"""
Ranker stability study: same retrieval model, ranker retrained with several seeds, legacy single-split
early stopping (cv_folds=0) vs K-fold mean-curve (cv_folds=3). Isolates RANKER variance.

    python scripts/ranker_seed_study.py --base-run 20261111-171751 --seeds 1 2 3 4 5

Runs live in models/<cat>/studies/ranker/ (never promoted, production untouched). Re-running skips
finished runs. Prints per-run results and mean +- std per mode, and writes study.md there.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

KEEP = ["data", "config.yaml", "two_tower.pt", "two_tower_trainonly.pt", "item_emb.npy", "faiss.index", "faiss_meta.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True, help="run whose retrieval model + data are reused")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--set", action="append", default=[], help="extra overrides, e.g. two_tower.text_encoder=tfidf")
    a = ap.parse_args()

    root = Path("models") / a.category
    base = root / "runs" / a.base_run
    ns = root / "studies" / "ranker"
    assert base.exists(), f"{base} not found"
    rows = []
    for mode, k in (("single_split", 0), ("cv3", 3)):
        for s in a.seeds:
            rid = f"{a.base_run}-{mode}-s{s}"
            d = ns / "runs" / rid
            if not (d / "metrics.json").exists():
                d.mkdir(parents=True, exist_ok=True)
                for f in KEEP:
                    src, dst = base / f, d / f
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    elif src.exists():
                        shutil.copy2(src, dst)
                cmd = [sys.executable, "-m", "pipeline.run", "--run-id", rid, "--steps", "train_ranker,evaluate",
                       "--set", "namespace=studies/ranker", "--set", f"ranker.seed={s}", "--set", f"ranker.cv_folds={k}",
                       "--set", "mlflow.experiment=recosphere-ranker-study"]
                for x in a.set:
                    cmd += ["--set", x]
                print(f"--- {rid}", flush=True)
                if subprocess.call(cmd) != 0:
                    raise SystemExit(f"{rid} failed")
            m = json.loads((d / "metrics.json").read_text())
            log = (d / "pipeline.log").read_text()
            best = re.findall(r"(?:best iteration|mean curve best) (\d+)", log)[-1]
            rows.append({"mode": mode, "seed": s, "best_iter": int(best), "ndcg10": m["ranker/all/ndcg@10"],
                         "recall100": m["ranker/all/recall@100"]})

    L = [f"# Ranker stability study (retrieval fixed: {a.base_run})", "",
         "| mode | seed | best iteration | test NDCG@10 | test Recall@100 |", "|---|---|---|---|---|"]
    L += [f"| {r['mode']} | {r['seed']} | {r['best_iter']} | {r['ndcg10']:.4f} | {r['recall100']:.4f} |" for r in rows]
    L += ["", "| mode | best iteration (min-max) | NDCG@10 mean +- std | Recall@100 mean +- std |", "|---|---|---|---|"]
    for mode in ("single_split", "cv3"):
        r = [x for x in rows if x["mode"] == mode]
        b, n, c = (np.array([x[key] for x in r]) for key in ("best_iter", "ndcg10", "recall100"))
        L.append(f"| {mode} | {b.min()}-{b.max()} | {n.mean():.4f} +- {n.std(ddof=1):.4f} | {c.mean():.4f} +- {c.std(ddof=1):.4f} |")
    old = {x["seed"]: x for x in rows if x["mode"] == "single_split"}
    wins = sum(x["ndcg10"] > old[x["seed"]]["ndcg10"] for x in rows if x["mode"] == "cv3")
    L += ["", f"cv3 beats single_split on NDCG@10 for {wins}/{len(a.seeds)} seeds (same seed = same user sample, paired)."]
    (ns / "study.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
