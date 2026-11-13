"""
A/B traffic simulator (Phase 8). We have no real users, so "users" replay their REAL future behaviour:

  user u arrives (random order = arrival over time) -> GET /recommend/u from the running API (real
  assignment, real models, real exposure log) -> a click model decides which shown items get clicked
  -> POST /feedback for each click (real click log).

Click model (position-biased, a simplified cascade/PBM model):
  P(examine position p) = 1 / p^decay                       (people look less at lower slots)
  P(click | examined)   = p_rel   if the item is in u's real test-window interactions (it is something u
                                   actually went on to engage with), else p_noise
So an arm that ranks the user's real future items higher gets more clicks. Honest limits: relevance is
"u interacted with it later", which ignores that showing an item can CREATE interest (counterfactual
bias), and every user makes one visit. The FRAMEWORK (assignment, logs, stats) is real; the effect
sizes are simulated and must be reported as such.

    python -m ab.simulate --api http://127.0.0.1:8000 --n-users 20000
    python -m ab.simulate --api http://127.0.0.1:8000 --n-users 5000 --interleave   # + interleaving data
"""
import argparse
import hashlib
import json
import os
import threading
import time
import http.client
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from ab.stats import team_draft_interleave


def ground_truth(gold: Path) -> tuple[list, dict]:
    """Users with history before the test window, and each one's NEW items in the test window."""
    hist = pl.concat([pl.read_parquet(gold / "train.parquet"), pl.read_parquet(gold / "val.parquet")])
    test = pl.read_parquet(gold / "test.parquet")
    um = pl.read_parquet(gold / "user_map.parquet")
    im = pl.read_parquet(gold / "item_map.parquet")
    new = test.join(hist.select("uid", "iid").unique(), on=["uid", "iid"], how="anti").unique(["uid", "iid"])
    new = new.join(um, on="uid").join(im, on="iid")
    truth = {u: set(items) for u, items in new.group_by("user_id").agg(pl.col("item_id")).iter_rows()}
    users = hist.select("uid").unique().join(um, on="uid")["user_id"].sort().to_list()
    return users, truth


def clicks_for(items, relevant, rng, decay, p_rel, p_noise):
    out = []
    for p, it in enumerate(items, start=1):
        if rng.random() < p ** -decay and rng.random() < (p_rel if it in relevant else p_noise):
            out.append(it)
    return out


_local = threading.local()


def _request(base, method, path, body=None, retries=4):
    """Keep-alive HTTP: one connection per thread, reused for every request. A new TCP connection per
    request exhausts ephemeral ports on Windows (WinError 10048) after ~16k requests."""
    u = urllib.parse.urlsplit(base)
    for attempt in range(retries):
        conn = getattr(_local, "conn", None)
        if conn is None:
            conn = _local.conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
        try:
            data = None if body is None else json.dumps(body)
            conn.request(method, path, body=data, headers={"content-type": "application/json"} if data else {})
            r = conn.getresponse()
            payload = r.read()
            if r.status >= 400:
                raise RuntimeError(f"HTTP {r.status} {path}: {payload[:200]!r}")
            return json.loads(payload)
        except (OSError, http.client.HTTPException):
            conn.close(); _local.conn = None
            if attempt == retries - 1:
                raise
            time.sleep(0.2 * 2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--gold", default=None, help="default: $RANKFORGE_DATA/gold/<category>")
    ap.add_argument("--category", default=os.environ.get("RANKFORGE_CATEGORY", "Video_Games"))
    ap.add_argument("--n-users", type=int, default=0, help="0 = every user with history (max power)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--p-rel", type=float, default=0.6)
    ap.add_argument("--p-noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interleave", action="store_true", help="also collect team-draft interleaving outcomes")
    ap.add_argument("--out", default="ab_runs/interleave.jsonl")
    a = ap.parse_args()

    gold = Path(a.gold) if a.gold else Path(os.environ.get("RANKFORGE_DATA", "data")) / "gold" / a.category
    users, truth = ground_truth(gold)
    rng = np.random.default_rng(a.seed)
    users = list(rng.permutation(users)[: a.n_users or None])  # random arrival order
    health = _request(a.api, "GET", "/health")
    exp = health.get("experiment")
    print(f"API model {health['model_run']} | experiment: {exp['name'] if exp else None} | "
          f"{len(users):,} users ({sum(u in truth for u in users):,} with test-window activity)")
    arms = [x["name"] for x in exp["arms"]] if exp else []
    if a.interleave:
        assert len(arms) == 2, "interleaving needs an active 2-arm experiment"
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        inter_f = open(a.out, "w", encoding="utf-8")
    lock, done, n_clicks, errors = threading.Lock(), [0], [0], [0]
    kw = dict(decay=a.decay, p_rel=a.p_rel, p_noise=a.p_noise)

    def visit(u):
        r = np.random.default_rng(int(hashlib.sha256(f"{a.seed}:{u}".encode()).hexdigest()[:12], 16))
        rel = truth.get(u, set())
        try:
            out = _request(a.api, "GET", f"/recommend/{urllib.parse.quote(u)}?k={a.k}")
            items = [x["item_id"] for x in out["items"]]
            cl = clicks_for(items, rel, r, **kw)
            for it in cl:
                _request(a.api, "POST", "/feedback", {"request_id": out["request_id"], "user_id": u, "item_id": it})
            rec = None
            if a.interleave:                                     # forced requests: excluded from the A/B logs
                q = urllib.parse.quote(u)
                la = [x["item_id"] for x in _request(a.api, "GET", f"/recommend/{q}?k={a.k}&force_arm={arms[0]}")["items"]]
                lb = [x["item_id"] for x in _request(a.api, "GET", f"/recommend/{q}?k={a.k}&force_arm={arms[1]}")["items"]]
                il, team = team_draft_interleave(la, lb, a.k, r)
                ci = set(clicks_for(il, rel, r, **kw))
                ca = sum(1 for it, t in zip(il, team) if it in ci and t == "A")
                cb = sum(1 for it, t in zip(il, team) if it in ci and t == "B")
                rec = json.dumps({"user_id": u, "a": arms[0], "b": arms[1], "clicks_a": ca, "clicks_b": cb})
            with lock:
                done[0] += 1; n_clicks[0] += len(cl)
                if rec:
                    inter_f.write(rec + "\n")
        except Exception as e:                                   # count, keep going
            with lock:
                errors[0] += 1
                if errors[0] <= 3:
                    print("error:", repr(e)[:200])

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for i, _ in enumerate(ex.map(visit, users), 1):
            if i % 5000 == 0:
                print(f"  {i:,} users  {time.time() - t0:.0f}s")
    if a.interleave:
        inter_f.close()
    print(f"done: {done[0]:,} visits, {n_clicks[0]:,} clicks, {errors[0]} errors in {time.time() - t0:.0f}s")
    if errors[0] > 0.01 * len(users):
        print(f"WARNING: {errors[0] / len(users):.1%} of visits failed -> the A/B sample is incomplete; fix and rerun "
              "(delete the exposure logs first)")


if __name__ == "__main__":
    main()
