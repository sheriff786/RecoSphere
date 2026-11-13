"""
A/B analysis (Phase 8): exposure log + click log -> decision report.

    python -m ab.analyze --logs ab_logs --experiment experiments/active.yaml --out ab_runs/<name>

Decision order: SRM -> guardrails -> power (no decision if MDE > --max-mde) -> significance.
Steps (in this order, because each one can invalidate the next):
  1. join      exposures (server: who saw what, which arm) + clicks (client) on request_id; drop forced (QA)
  2. SRM       user counts per arm vs configured split. Mismatch -> STOP: results are not trustworthy
  3. A/A       split the control arm in two 500x: false-positive rate must be ~5% (calibrates the stats)
  4. metrics   per USER (the randomization unit): clicks, any-click (conversion)
  5. CUPED     same comparison with pre-experiment activity as covariate -> narrower CI, same effect
  6. guardrails latency p95, fallback/error share, catalog coverage: a "win" that breaks these is not a win
  7. power     MDE with this many users; users needed for a 2% / 5% lift
  8. decision  ship / don't ship / inconclusive (+ peeking demo, + interleaving if collected)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ab import stats as S


def load_logs(d: Path, experiment: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    def read(pattern):
        files = sorted(d.glob(pattern))
        return pl.concat([pl.read_ndjson(f, infer_schema_length=None) for f in files if f.stat().st_size],
                         how="diagonal_relaxed") if files else pl.DataFrame()
    ex, cl = read("exposures-*.jsonl"), read("clicks-*.jsonl")
    ex = ex.filter((pl.col("experiment") == experiment) & ~pl.col("forced"))
    if cl.height:
        cl = cl.filter(pl.col("experiment") == experiment)
    return ex, cl


def covariate(gold: Path, users: pl.Series) -> np.ndarray:
    """Pre-experiment activity = interactions in the val window (the year before the experiment) + history
    length. Measured BEFORE assignment -> cannot be affected by the treatment (CUPED requirement)."""
    um = pl.read_parquet(gold / "user_map.parquet")
    va = pl.read_parquet(gold / "val.parquet").group_by("uid").len("pre")
    tr = pl.read_parquet(gold / "train.parquet").group_by("uid").len("old")
    x = (pl.DataFrame({"user_id": users}).join(um, on="user_id", how="left")
         .join(va, on="uid", how="left").join(tr, on="uid", how="left").fill_null(0))
    return np.log1p(x["pre"].to_numpy()) + 0.5 * np.log1p(x["old"].to_numpy())


def fmt_pct(x):
    return f"{x:+.2%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="ab_logs")
    ap.add_argument("--experiment", default="experiments/active.yaml")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--interleave", default=None, help="interleave.jsonl from ab.simulate --interleave")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-mde", type=float, default=0.10,
                    help="do not decide if the test cannot detect a relative effect this small (underpowered)")
    a = ap.parse_args()

    import os
    gold = Path(a.gold) if a.gold else Path(os.environ.get("RANKFORGE_DATA", "data")) / "gold" / a.category
    exp = yaml.safe_load(Path(a.experiment).read_text())
    share = {x["name"]: float(x["traffic"]) for x in exp["arms"]}
    treat = [x for x in share if x != "control"]
    out = Path(a.out or f"ab_runs/{exp['name']}"); out.mkdir(parents=True, exist_ok=True)
    R = {"experiment": exp["name"], "salt": exp["salt"], "arms": exp["arms"]}
    L = [f"# A/B report: {exp['name']}", "",
         "> Simulated users (ab/simulate.py click model on real future interactions). The framework is real;",
         "> effect sizes are simulation results, not production numbers.", ""]

    # 1. join ---------------------------------------------------------------------------------------
    ex, cl = load_logs(Path(a.logs), exp["name"])
    assert ex.height, f"no exposures for experiment {exp['name']} in {a.logs}"
    n_cl = cl.group_by("request_id").len("clicks") if cl.height else pl.DataFrame({"request_id": [], "clicks": []},
                                                                                     schema={"request_id": pl.String, "clicks": pl.UInt32})
    req = ex.join(n_cl, on="request_id", how="left").with_columns(pl.col("clicks").fill_null(0))
    orphan = cl.join(ex.select("request_id"), on="request_id", how="anti").height if cl.height else 0
    users = (req.group_by("user_id").agg(pl.col("arm").n_unique().alias("n_arms"), pl.col("arm").first(),
                                         pl.len().alias("requests"), pl.col("clicks").sum())
             .with_columns((pl.col("clicks") > 0).cast(pl.Float64).alias("converted")))
    switched = users.filter(pl.col("n_arms") > 1).height
    L += ["## 1. Data", "", f"- exposures: {ex.height:,} (forced/QA requests excluded) | clicks: {cl.height:,} | "
          f"users: {users.height:,}", f"- clicks without a matching exposure: {orphan} | users seen in >1 arm: {switched} "
          "(both must be 0)", ""]
    R["data"] = {"exposures": ex.height, "clicks": cl.height, "users": users.height, "orphan_clicks": orphan,
                 "users_in_multiple_arms": switched}

    # 2. SRM ----------------------------------------------------------------------------------------
    counts = {k: users.filter(pl.col("arm") == k).height for k in share}
    srm = S.srm_test(counts, share)
    R["srm"] = srm
    L += ["## 2. Sample ratio mismatch (SRM)", "", f"- users per arm: {counts} | expected {srm['expected']}",
          f"- chi² = {srm['chi2']:.2f}, p = {srm['p_value']:.4f} -> **{'SRM! results NOT trustworthy' if srm['srm'] else 'OK'}**", ""]

    # 3. A/A on control -----------------------------------------------------------------------------
    ctl = users.filter(pl.col("arm") == "control")
    aa = S.aa_false_positive_rate(ctl["user_id"].to_list(), ctl["clicks"].to_numpy(), n_sims=500, alpha=a.alpha)
    R["aa"] = aa
    L += ["## 3. A/A calibration (control split in two, 500 random splits)", "",
          f"- false-positive rate = {aa['false_positive_rate']:.1%} (should be ~{a.alpha:.0%}); "
          f"p-values uniform? KS p = {aa['p_uniform_ks_p']:.2f}", ""]

    # 4-5. metrics + CUPED ---------------------------------------------------------------------------
    x = covariate(gold, users["user_id"])
    users = users.with_columns(pl.Series("x_pre", x))
    R["metrics"] = {}
    L += ["## 4. Primary metrics (per user)", "",
          "| arm vs control | metric | control | treatment | lift | 95% CI (lift) | p | CUPED lift CI | CUPED var. reduction |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in treat:
        sub = users.filter(pl.col("arm").is_in(["control", t]))
        g = (sub["arm"] == t).to_numpy()
        for metric in ("clicks", "converted"):
            y = sub[metric].to_numpy().astype(float)
            raw = S.compare_means(y[g], y[~g], a.alpha)
            adj, theta, red = S.cuped(y, sub["x_pre"].to_numpy())
            cup = S.compare_means(adj[g], adj[~g], a.alpha)
            # relative CI for CUPED: same denominator (control mean of raw y)
            rel_ci = (cup["ci"][0] / raw["mean_c"], cup["ci"][1] / raw["mean_c"])
            R["metrics"][f"{t}/{metric}"] = {"raw": raw, "cuped": cup, "cuped_rel_ci": rel_ci, "theta": theta,
                                             "variance_reduction": red}
            L.append(f"| {t} | {metric} | {raw['mean_c']:.4f} | {raw['mean_t']:.4f} | {fmt_pct(raw['rel_lift'])} | "
                     f"[{fmt_pct(raw['rel_ci'][0])}, {fmt_pct(raw['rel_ci'][1])}] | {raw['p_value']:.4f} | "
                     f"[{fmt_pct(rel_ci[0])}, {fmt_pct(rel_ci[1])}] | {red:.1%} |")
    L.append("")

    # 6. guardrails ---------------------------------------------------------------------------------
    n_eq = int(ex.group_by("arm").len()["len"].min())                 # equal-size samples -> fair coverage
    cov = {arm: ex.filter(pl.col("arm") == arm).sample(n_eq, seed=0)["items"].explode().n_unique()
           for arm in share}
    has_new = "n_new" in ex.columns
    gr = (ex.group_by("arm").agg(pl.col("latency_ms").quantile(0.5).alias("p50_ms"),
                                  pl.col("latency_ms").quantile(0.95).alias("p95_ms"),
                                  (pl.col("source") == "fallback").mean().alias("fallback_share"),
                                  (pl.col("source") == "fallback_error").mean().alias("error_share"),
                                  ((pl.col("n_new") / pl.col("k")).mean() if has_new else pl.lit(None)).alias("new_item_share"),
                                  pl.len().alias("impressions"))
          .with_columns(pl.col("arm").replace_strict(cov, default=None).alias("distinct_items"))
          .sort("arm"))
    R["guardrails"] = gr.to_dicts()
    c_row = gr.filter(pl.col("arm") == "control").to_dicts()[0]
    L += ["## 5. Guardrails", "", f"| arm | p50 ms | p95 ms | fallback share | error share | new-item share | "
          f"distinct items in {n_eq:,} impressions |", "|---|---|---|---|---|---|---|"]
    guard_fail = []
    for r in gr.to_dicts():
        ns = "n/a" if r["new_item_share"] is None else f"{r['new_item_share']:.1%}"
        L.append(f"| {r['arm']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['fallback_share']:.2%} | "
                 f"{r['error_share']:.3%} | {ns} | {r['distinct_items']:,} |")
        if r["arm"] != "control":
            if r["p95_ms"] > 1.25 * c_row["p95_ms"] and r["p95_ms"] > 50:
                guard_fail.append(f"{r['arm']}: p95 latency {r['p95_ms']:.0f} ms vs control {c_row['p95_ms']:.0f} ms")
            if r["error_share"] > max(0.001, 2 * c_row["error_share"]):
                guard_fail.append(f"{r['arm']}: error share {r['error_share']:.3%}")
            if r["distinct_items"] < 0.95 * c_row["distinct_items"]:
                guard_fail.append(f"{r['arm']}: catalog coverage {r['distinct_items']:,} vs control {c_row['distinct_items']:,} "
                                  f"({r['distinct_items'] / c_row['distinct_items'] - 1:+.0%})")
            if has_new and c_row["new_item_share"] and r["new_item_share"] < 0.5 * c_row["new_item_share"]:
                guard_fail.append(f"{r['arm']}: new-item share {r['new_item_share']:.1%} vs control {c_row['new_item_share']:.1%}")
    L += ["", "Rules: p95 latency within +25% of control (or under 50 ms); error share <= max(0.1%, 2x control); "
          "catalog coverage (distinct items on equal-size samples) at most 5% below control; new-item share at least "
          "half of control. Clicks are short-term: coverage and new-item exposure protect the catalog long-term.", ""]
    R["guardrail_failures"] = guard_fail

    # 7. power --------------------------------------------------------------------------------------
    y_c = ctl["clicks"].to_numpy().astype(float)
    n_arm = int(np.mean(list(counts.values())))
    sd, base = y_c.std(ddof=1), y_c.mean()
    red = R["metrics"][f"{treat[0]}/clicks"]["variance_reduction"]
    pw = {"baseline_clicks_per_user": base, "sd": sd, "users_per_arm": n_arm,
          "mde_rel": S.mde(sd, n_arm) / base, "mde_rel_cuped": S.mde(sd * np.sqrt(1 - red), n_arm) / base,
          "users_per_arm_for_2pct": S.sample_size(sd, 0.02 * base), "users_per_arm_for_5pct": S.sample_size(sd, 0.05 * base),
          "users_per_arm_for_2pct_cuped": S.sample_size(sd * np.sqrt(1 - red), 0.02 * base)}
    R["power"] = pw
    L += ["## 6. Power (clicks per user, alpha 5%, power 80%)", "",
          f"- baseline {base:.4f} clicks/user, sd {sd:.3f}, {n_arm:,} users per arm",
          f"- minimum detectable effect: **{pw['mde_rel']:.1%}** relative (with CUPED: {pw['mde_rel_cuped']:.1%})",
          f"- users per arm needed for a 2% lift: {pw['users_per_arm_for_2pct']:,} (CUPED: {pw['users_per_arm_for_2pct_cuped']:,}); "
          f"for 5%: {pw['users_per_arm_for_5pct']:,}", ""]

    # peeking demo ------------------------------------------------------------------------------------
    pk = S.peeking_false_positive(y_c, n_looks=20, n_sims=500, alpha=a.alpha)
    R["peeking"] = pk
    L += ["## 7. Why we do not peek", "",
          f"A/A on control traffic, checking the p-value 20 times as users arrive and stopping at the first p < 5%: "
          f"false-positive rate **{pk['fpr_peeking']:.1%}** vs {pk['fpr_fixed_horizon']:.1%} with one look at the planned end. "
          "Fix the sample size up front (section 6), or use a sequential test built for repeated looks.", ""]

    # interleaving -----------------------------------------------------------------------------------
    if a.interleave and Path(a.interleave).exists():
        il = pl.read_ndjson(a.interleave)
        wb = (il["clicks_b"] > il["clicks_a"]).to_numpy(); wa = (il["clicks_a"] > il["clicks_b"]).to_numpy()
        it = S.interleaving_test(wb, wa)
        # sensitivity: how often does each method reach p<0.05 on random subsets of n users?
        rng = np.random.default_rng(0)
        sub_t = users.filter(pl.col("arm") == treat[0])["clicks"].to_numpy().astype(float)
        sizes = [n for n in (250, 500, 1000, 2000, 4000, 8000) if n <= min(len(il), len(sub_t), len(y_c))]
        sens = []
        for n in sizes:
            hit_ab = hit_il = 0
            for _ in range(200):
                hit_ab += S.compare_means(rng.choice(sub_t, n // 2), rng.choice(y_c, n // 2))["p_value"] < a.alpha
                idx = rng.choice(len(il), n)
                hit_il += S.interleaving_test(wb[idx], wa[idx])["p_value"] < a.alpha
            sens.append({"users_total": n, "power_ab": hit_ab / 200, "power_interleaving": hit_il / 200})
        R["interleaving"] = {**it, "n_users": len(il), "sensitivity": sens}
        L += ["## 8. Interleaving (same users see ONE list mixed from both rankers)", "",
              f"- {len(il):,} users: treatment ('{il['b'][0]}') won {it['users_b_won']:,}, control won {it['users_a_won']:,} "
              f"(ties ignored) -> delta = {it['delta']:+.3f}, p = {it['p_value']:.4g}",
              "", "| users | power A/B | power interleaving |", "|---|---|---|"]
        L += [f"| {s['users_total']:,} | {s['power_ab']:.0%} | {s['power_interleaving']:.0%} |" for s in sens]
        L += ["", "Interleaving compares both rankers on the SAME user and the same visit, so user-to-user variance "
              "cancels -> it needs far fewer users. It only answers 'which ranking do users prefer', not business "
              "metrics (revenue, retention), so it is a fast pre-filter before a real A/B test.", ""]

    # 8. decision ------------------------------------------------------------------------------------
    decisions = {}
    for t in treat:
        m = R["metrics"][f"{t}/clicks"]
        lo, hi = m["cuped_rel_ci"]
        if srm["srm"]:
            d = "INVALID (SRM): fix assignment/logging, rerun"
        elif any(f.startswith(t) for f in guard_fail):
            d = "DON'T SHIP: guardrail failed"
        elif pw["mde_rel_cuped"] > a.max_mde:
            d = (f"NO DECISION (underpowered): MDE {pw['mde_rel_cuped']:.0%} > {a.max_mde:.0%}. Observed lift "
                 f"{fmt_pct(m['raw']['rel_lift'])} [{fmt_pct(lo)}, {fmt_pct(hi)}]. At this power a 'significant' result "
                 f"usually overstates the true effect (type-M error); need ~{S.sample_size(sd, a.max_mde * base):,} users per arm")
        elif lo > 0:
            d = "SHIP: primary metric up, CI excludes 0, guardrails OK"
        elif hi < 0:
            d = "DON'T SHIP: primary metric significantly down"
        else:
            d = f"INCONCLUSIVE: CI [{fmt_pct(lo)}, {fmt_pct(hi)}] includes 0 -> effect smaller than the MDE; need more users or accept no change"
        decisions[t] = d
    R["decision"] = decisions
    L = L[:4] + ["## Decision", ""] + [f"- **{t}**: {d}" for t, d in decisions.items()] + [""] + L[4:]
    if guard_fail:
        L += ["Guardrail failures: " + "; ".join(guard_fail), ""]

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else list(o)))
    users.write_parquet(out / "per_user.parquet")
    print("\n".join(L))
    print(f"\nwritten: {out / 'report.md'}")


if __name__ == "__main__":
    main()
