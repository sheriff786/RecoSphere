"""
Time-travel replay (Phase 7e): "what if this pipeline had been running every N days in the past?"

    python -m pipeline.replay --name q90 --start 2022-01-01 --end 2023-04-01 --every-days 90
    python -m pipeline.replay --name q90 ... --set two_tower.epochs=4      # any pipeline override

For each cutoff c (start, start+N, ...):
  1. full pipeline in cutoff mode: train < c-365d <= val < c <= test < c+N   ("today" = c)
     -> gate vs the replay's own production -> promote (exit 2 = gate said no, replay continues)
     Runs live in models/<cat>/replay/<name>/ with their own production.json: the REAL production
     pointer (what the API serves) is never touched.
  2. decay matrix: every model i is evaluated on every LATER window j (stale weights, fresh user
     history = what serving does when you do not retrain). Cached per pair.

Outputs (models/<cat>/replay/<name>/):
  timeline.md/json   per cutoff: data size, metrics, gate decision
  decay.csv/md       metric(model i, window j)
  decay.png          NDCG@10 vs model age
  policies.md        average quality of: never retrain / retrain every k periods / gated retrain (what
                     the pipeline actually did) / always retrain -> evidence for the retrain schedule
Re-running the same command resumes: finished runs and evaluated pairs are skipped.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from pipeline.config import Paths, load_config
from pipeline.data import load_run_data

METRICS = ["ranker/all/ndcg@10", "ranker/all/recall@100", "candidates/recall_union", "recent_pop/all/ndcg@10"]


def cutoffs(start, end, every):
    c, out = datetime.fromisoformat(start), []
    while c <= datetime.fromisoformat(end):
        out.append(c.date().isoformat())
        c += timedelta(days=every)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--every-days", type=int, default=90)
    ap.add_argument("--val-days", type=int, default=365)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-decay", action="store_true")
    a = ap.parse_args()

    ns = f"replay/{a.name}"
    base = [f"namespace={ns}", "split.mode=cutoff", f"split.val_days={a.val_days}",
            f"split.test_days={a.every_days}", f"mlflow.experiment=recosphere-replay-{a.name}"]
    cfg = load_config(None, base + a.set)
    paths = Paths(cfg)
    if "replay" not in paths.models.parts:                     # never touch the real production pointer
        raise SystemExit(f"replay would write to {paths.models} (not a replay namespace). "
                         "pipeline/config.py is too old: it must support the 'namespace' key.")
    cuts = cutoffs(a.start, a.end, a.every_days)
    print(f"replay '{a.name}': {len(cuts)} cutoffs {cuts[0]} .. {cuts[-1]} every {a.every_days}d -> {paths.models}")

    # 1. one pipeline run per cutoff ----------------------------------------------------------------
    timeline = []
    for c in cuts:
        rid = f"cut-{c}"
        rdir = paths.runs / rid
        if not (rdir / "gate_report.json").exists():
            cmd = [sys.executable, "-m", "pipeline.run", "--run-id", rid]
            for s in base + a.set + [f"split.cutoff={c}"]:
                cmd += ["--set", s]
            print(f"\n=== cutoff {c} ===", flush=True)
            code = subprocess.call(cmd)
            if code not in (0, 2):
                raise SystemExit(f"pipeline failed at cutoff {c} (exit {code})")
        man = json.loads((rdir / "data" / "manifest.json").read_text())
        m = json.loads((rdir / "metrics.json").read_text())
        gate = json.loads((rdir / "gate_report.json").read_text())
        timeline.append({"cutoff": c, "run_id": rid, "train_rows": man["rows"]["train"],
                         "test_rows": man["rows"]["test"], "test_users": m["eval/n_users"],
                         **{k: m[k] for k in METRICS}, "gate_passed": gate["passed"],
                         "failed_checks": [x["check"] for x in gate["checks"] if not x["passed"]]})
    hist = [json.loads(x) for x in paths.production_history.read_text().splitlines() if x.strip()]
    promoted = {h["run_id"] for h in hist}
    for t in timeline:
        t["promoted"] = t["run_id"] in promoted
    (paths.models / "timeline.json").write_text(json.dumps(timeline, indent=2))
    L = [f"# Replay '{a.name}': timeline", "", "| cutoff | train rows | test users | ranker NDCG@10 | Recall@100 | "
         "cand. recall | recent-pop NDCG@10 | gate | promoted |", "|---|---|---|---|---|---|---|---|---|"]
    for t in timeline:
        L.append(f"| {t['cutoff']} | {t['train_rows']:,} | {t['test_users']:,} | {t['ranker/all/ndcg@10']:.4f} | "
                 f"{t['ranker/all/recall@100']:.4f} | {t['candidates/recall_union']:.4f} | {t['recent_pop/all/ndcg@10']:.4f} | "
                 f"{'PASS' if t['gate_passed'] else 'FAIL: ' + '; '.join(t['failed_checks'])} | {'yes' if t['promoted'] else 'no'} |")
    L += ["", "Absolute numbers across rows are NOT comparable (each row is a different test window, different users)."
          " Compare models only within the same window: decay.md.", ""]
    (paths.models / "timeline.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if a.no_decay:
        return

    # 2. decay matrix ------------------------------------------------------------------------------
    from pipeline.steps import evaluate_models
    rows = []
    for j, cj in enumerate(cuts):
        rdj = paths.runs / f"cut-{cj}"
        rd = None
        for i, ci in enumerate(cuts[: j + 1]):
            cache = rdj / f"decay_from_cut-{ci}.json"
            if i == j:
                m = json.loads((rdj / "metrics.json").read_text())
            elif cache.exists():
                m = json.loads(cache.read_text())
            else:
                rd = rd or load_run_data(rdj, paths.gold)
                m = evaluate_models(paths.runs / f"cut-{ci}", rd, cfg["serving"]["new_item_slots"], print)
                cache.write_text(json.dumps(m, indent=2))
            age = (datetime.fromisoformat(cj) - datetime.fromisoformat(ci)).days
            rows.append({"model": ci, "window": cj, "age_days": age, **{k: m[k] for k in METRICS}})
    D = pl.DataFrame(rows)
    fresh = D.filter(pl.col("age_days") == 0).select("window", pl.col("ranker/all/ndcg@10").alias("fresh_ndcg"),
                                                      pl.col("ranker/all/recall@100").alias("fresh_recall"))
    D = D.join(fresh, on="window").with_columns(
        (pl.col("ranker/all/ndcg@10") / pl.col("fresh_ndcg") - 1).alias("ndcg_vs_fresh"),
        (pl.col("ranker/all/recall@100") / pl.col("fresh_recall") - 1).alias("recall_vs_fresh"))
    D.write_csv(paths.models / "decay.csv")

    piv = D.pivot(on="window", index="model", values="ranker/all/ndcg@10", aggregate_function="first").sort("model")
    L = [f"# Replay '{a.name}': model decay", "", "Ranker NDCG@10 of model (row) on test window (column). Diagonal = "
         "fresh model. Read DOWN a column: same users, older and older models.", "",
         "| model \\ window | " + " | ".join(cuts) + " |", "|---|" + "---|" * len(cuts)]
    for r in piv.iter_rows(named=True):
        L.append(f"| {r['model']} | " + " | ".join("" if r.get(c) is None else f"{r[c]:.4f}" for c in cuts) + " |")
    by_age = (D.filter(pl.col("age_days") > 0).group_by("age_days")
              .agg(pl.col("ndcg_vs_fresh").mean(), pl.col("recall_vs_fresh").mean(), pl.len().alias("pairs")).sort("age_days"))
    L += ["", "## Stale model vs fresh model, by age (same window)", "", "| model age (days) | NDCG@10 vs fresh | "
          "Recall@100 vs fresh | pairs |", "|---|---|---|---|"]
    L += [f"| {r['age_days']} | {r['ndcg_vs_fresh']:+.1%} | {r['recall_vs_fresh']:+.1%} | {r['pairs']} |"
          for r in by_age.iter_rows(named=True)]
    (paths.models / "decay.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # 3. retrain policies --------------------------------------------------------------------------
    look = {(r["model"], r["window"]): r["ranker/all/ndcg@10"] for r in rows}
    prom = [t["cutoff"] for t in timeline if t["promoted"]]

    def serve(policy):
        vals = []
        for j, cj in enumerate(cuts):
            if policy == "gated (what the pipeline did)":
                m = max([c for c in prom if c <= cj], default=cuts[0])
            elif policy.startswith("every"):
                k = int(policy.split()[1])
                m = cuts[(j // k) * k]
            elif policy == "never retrain":
                m = cuts[0]
            else:
                m = cj
            vals.append(look[(m, cj)])
        return float(np.mean(vals)), vals

    pols = ["always retrain"] + [f"every {k} periods" for k in (2, 3) if k < len(cuts)] + \
           ["gated (what the pipeline did)", "never retrain"]
    best = serve("always retrain")[0]
    L = [f"# Replay '{a.name}': retrain policies", "",
         f"Mean ranker NDCG@10 over {len(cuts)} windows of {a.every_days} days if production follows each policy.", "",
         "| policy | mean NDCG@10 | vs always retrain | per window |", "|---|---|---|---|"]
    for p in pols:
        mean, vals = serve(p)
        L.append(f"| {p} | {mean:.4f} | {mean / best - 1:+.1%} | {' '.join(f'{v:.4f}' for v in vals)} |")
    L += ["", "Read with the noise in mind: one window has only a few thousand test users and every model is one seed, "
          "so differences of a few % between policies are within noise (see the gate CIs in each run's "
          "gate_report.json). Only a consistent gap (e.g. 'never retrain' losing in most windows) is evidence.",
          "Cost side (not in the table): one retrain = one pipeline run (see MLflow time/* metrics). "
          "Pick the cheapest policy whose loss vs 'always' is below what you can detect/care about.", ""]
    (paths.models / "policies.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c in cuts:
            s = D.filter(pl.col("model") == c).sort("window")
            ax[0].plot(s["window"].to_list(), s["ranker/all/ndcg@10"].to_list(), marker="o", label=f"model {c}")
        ax[0].set_title("NDCG@10 per test window (one line per model)"); ax[0].tick_params(axis="x", rotation=45)
        ax[0].legend(fontsize=7)
        ax[1].plot(by_age["age_days"].to_list(), (by_age["ndcg_vs_fresh"] * 100).to_list(), marker="o", label="NDCG@10")
        ax[1].plot(by_age["age_days"].to_list(), (by_age["recall_vs_fresh"] * 100).to_list(), marker="s", label="Recall@100")
        ax[1].axhline(0, color="grey", lw=0.8); ax[1].set_xlabel("model age (days)"); ax[1].set_ylabel("% vs fresh model")
        ax[1].set_title("Stale vs fresh model, same window"); ax[1].legend()
        fig.tight_layout(); fig.savefig(paths.models / "decay.png", dpi=120)
    except Exception as e:                                                     # plot is optional
        print("plot skipped:", e)
    print(f"\nreplay outputs in {paths.models}")


if __name__ == "__main__":
    main()
