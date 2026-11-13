"""
Phase 1 runner. Protocol (isse REPORT.md mein bhi likho):
  1. Hyperparams (EASE λ, ItemKNN k) VAL par tune: fit(train) -> eval(val)
  2. Best config ko train+val par refit -> eval(test). Test sirf ek baar dekha jaata hai.

Usage:
    python run_baselines.py --category Video_Games
    python run_baselines.py --category Video_Games --ease-max-items 10000   # kam RAM
Output: results/<category>/baselines.csv  aur  baselines.md (REPORT mein paste karo)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from baselines.models import EASE, ItemKNN, Popularity
from eval.metrics import build_matrix, evaluate

KS = (10, 50, 100)
SHOW = ["all/recall@50", "all/recall@100", "all/ndcg@10", "warm/recall@100", "cold/recall@100"]


def recent_slice(df: pl.DataFrame, frac: float) -> pl.DataFrame:
    t = df["ts"].quantile(1 - frac, interpolation="lower")
    return df.filter(pl.col("ts") >= t)


def fit_model(cfg, hist_df, n_users, n_items):
    X = build_matrix(hist_df, n_users, n_items)
    kind, p = cfg
    t0 = time.time()
    if kind == "pop":
        m = Popularity(p)
        recent = build_matrix(recent_slice(hist_df, p), n_users, n_items) if p else None
        m.fit(X, recent)
    elif kind == "knn":
        m = ItemKNN(k=p).fit(X)
    else:
        m = EASE(lam=p, max_items=ARGS.ease_max_items).fit(X)
    return m, X, time.time() - t0


def run(cfg, hist_df, eval_df, n_users, n_items):
    m, X, fit_s = fit_model(cfg, hist_df, n_users, n_items)
    t0 = time.time()
    r = evaluate(m.score, X, eval_df, hist_df["iid"].unique().to_numpy(), ks=KS)
    r.update(model=m.name, fit_s=round(fit_s, 1), eval_s=round(time.time() - t0, 1))
    return r


def main():
    d = Path(ARGS.root) / ARGS.category
    train, val, test = (pl.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test"))
    n_users = pl.read_parquet(d / "user_map.parquet").height
    n_items = pl.read_parquet(d / "item_map.parquet").height
    train_val = pl.concat([train, val])

    families = {
        "pop": [None, 0.1,0.05, 0.02],                 # all-time vs recent-10%
        "knn": ARGS.knn_k,
        "ease": ARGS.ease_lams,
    }
    final = []
    for fam, grid in families.items():
        print(f"\n== {fam}: tuning on VAL ==")
        best, best_cfg = -1, None
        for p in grid:
            r = run((fam, p), train, val, n_users, n_items)
            print(f"  {r['model']:<22} val ndcg@10={r['all/ndcg@10']:.4f}  recall@100={r['all/recall@100']:.4f}"
                  f"  fit={r['fit_s']}s")
            # popularity ke dono variants test par report karo (comparison khud ek finding hai)
            if fam == "pop":
                final.append(("pop", p))
            elif r["all/ndcg@10"] > best:
                best, best_cfg = r["all/ndcg@10"], (fam, p)
        if best_cfg:
            final.append(best_cfg)

    print("\n== FINAL: refit on train+val, eval on TEST (once) ==")
    rows = []
    for cfg in final:
        r = run(cfg, train_val, test, n_users, n_items)
        rows.append(r)
        print(f"  {r['model']:<22} " + "  ".join(f"{k}={r[k]:.4f}" for k in SHOW))

    out = Path("results") / ARGS.category
    out.mkdir(parents=True, exist_ok=True)
    res = pl.DataFrame(rows)
    res.write_csv(out / "baselines.csv")
    meta = rows[0]
    md = [f"Test users: {meta['n_eval_users']:,} | mean cold-target share: {meta['mean_cold_target_share']:.1%}\n",
          "| model | " + " | ".join(SHOW) + " | fit (s) |",
          "|---" * (len(SHOW) + 2) + "|"]
    for r in rows:
        md.append(f"| {r['model']} | " + " | ".join(f"{r[k]:.4f}" for k in SHOW) + f" | {r['fit_s']} |")
    (out / "baselines.md").write_text("\n".join(md) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--ease-lams", type=float, nargs="+", default=[100.0, 500.0, 2000.0])
    ap.add_argument("--ease-max-items", type=int, default=20000)
    ap.add_argument("--knn-k", type=int, nargs="+", default=[50, 200])
    ARGS = ap.parse_args()
    main()
