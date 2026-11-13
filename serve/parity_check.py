"""
Training-serving parity check — two tests.

TEST A  (the real skew test)
  Give the API's feature code (feature store -> UserBlock -> assemble) the SAME candidates the notebook
  path produced. All 22 features must match the notebook's features (float tolerance 1e-5).
  If this fails, the API computes features differently from training -> real bug.

TEST B  (end-to-end agreement)
  Full API path (FAISS + store + features + ranker) vs notebook path. Candidate sets and top-10 order
  are compared. Small disagreements come from float noise: two items whose two-tower scores differ by
  ~1e-7 can swap rank, which changes the tt_rank feature, and LightGBM (a tree model) can jump on that.
  Every top-10 mismatch is checked: is it explained by such a near-tie?

PASS = Test A passes AND >= 99% same candidate sets AND every top-10 mismatch is a near-tie.

Run from the repo root:
    python -m serve.parity_check --n 300
    REDIS_URL=redis://localhost:6379/0 python -m serve.parity_check --n 300
"""
import argparse
import sys
import time

import numpy as np
import polars as pl

from serve.config import ServeConfig          # first: sets the Windows OpenMP flag before torch/faiss load
from ranking.pipeline import FEATS, assemble, features, gen_candidates
from serve.recommender import Recommender

FEATURE_TOL = 1e-5
NEAR_TIE = 1e-5          # two-tower score gap below this = float-noise tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rec = Recommender(ServeConfig.from_env())
    test = pl.read_parquet(rec.cfg.gold / "test.parquet")
    with_hist = set(rec.hist["uid"].unique().to_list())
    cand_users = sorted(set(test["uid"].unique().to_list()) & with_hist)
    rng = np.random.default_rng(a.seed)
    sample = np.sort(rng.choice(cand_users, size=min(a.n, len(cand_users)), replace=False))

    t0 = time.time()
    c = gen_candidates(rec.ctx, rec.model, rec.hist, test.filter(pl.col("uid").is_in(sample.tolist())),
                       rec.t_ref, V=rec.V)                     # same item vectors as the API
    F = features(rec.ctx, c, rec.hist)
    batch_scores = rec.booster.predict(F).astype(np.float32)
    starts = np.searchsorted(c["u"], c["users"]); ends = np.append(starts[1:], len(c["u"]))
    print(f"notebook path: {len(c['users'])} users ({time.time() - t0:.1f}s) | store backend: {rec.store.backend}")

    names = dict(zip(rec.user_map["uid"].to_list(), rec.user_map["user_id"].to_list()))
    feat_diff = {f: 0.0 for f in FEATS}; nan_mis = 0
    same_set = same_top10 = explained = 0
    unexplained = []
    for u, s0, s1 in zip(c["users"], starts, ends):
        _, r = rec.store.get(names[int(u)])
        b_items, b_scores = c["i"][s0:s1], batch_scores[s0:s1]

        # ---- TEST A: same candidates -> API feature code must give the same features ----
        Xa = assemble(rec.ctx, rec.items, rec.user_block(r), np.zeros(len(b_items), np.int64), b_items,
                      c["tt_score"][s0:s1], c["tt_rank"][s0:s1], c["pop_rank"][s0:s1])
        Fb = F.iloc[s0:s1]
        for f in FEATS:
            bv, av = Fb[f].to_numpy(), Xa[f].to_numpy()
            nan_mis += int((np.isnan(bv) != np.isnan(av)).sum())
            both = ~np.isnan(bv) & ~np.isnan(av)
            if both.any():
                feat_diff[f] = max(feat_diff[f], float(np.abs(bv[both] - av[both]).max()))

        # ---- TEST B: full API path ----
        s_items, s_scores, s_X = rec.score_candidates(r, {}, return_features=True)
        same_set += set(b_items.tolist()) == set(s_items.tolist())
        b_top = b_items[np.argsort(-b_scores, kind="stable")][:10]
        s_top = s_items[np.argsort(-s_scores, kind="stable")][:10]
        if np.array_equal(b_top, s_top):
            same_top10 += 1
            continue
        # A mismatch is "explained" only if (1) every feature except tt_rank matches within float noise and
        # (2) tt_rank differs only for items that sit in a near-tie with a neighbour.
        pos = {int(x): j for j, x in enumerate(s_items)}
        idx = np.array([pos.get(int(x), -1) for x in b_items]); ok = idx >= 0
        others_ok = True
        for f in FEATS:
            if f == "tt_rank":
                continue
            bv, sv = Fb[f].to_numpy()[ok], s_X[f].to_numpy()[idx[ok]]
            both = ~np.isnan(bv) & ~np.isnan(sv)
            if (np.isnan(bv) != np.isnan(sv)).any() or (both.any() and np.abs(bv[both] - sv[both]).max() > 1e-4):
                others_ok = False; break
        tt = c["tt_score"][s0:s1]; has = ~np.isnan(c["tt_rank"][s0:s1])
        order = np.argsort(-tt[has]); srt = tt[has][order]
        near = np.zeros(len(srt), bool)
        gap = np.abs(np.diff(srt)) < NEAR_TIE
        near[:-1] |= gap; near[1:] |= gap
        near_items = set(b_items[has][order][near].tolist())
        br, sr = Fb["tt_rank"].to_numpy()[ok], s_X["tt_rank"].to_numpy()[idx[ok]]
        diff_items = set(b_items[ok][~((br == sr) | (np.isnan(br) & np.isnan(sr)))].tolist())
        if others_ok and diff_items <= near_items:
            explained += 1
        else:
            unexplained.append(int(u))

    n = len(c["users"])
    worst = sorted(feat_diff.items(), key=lambda kv: -kv[1])[:4]
    test_a = all(v < FEATURE_TOL for v in feat_diff.values()) and nan_mis == 0
    print("\nTEST A  feature code (same candidates in, same features out)")
    print(f"  max |diff| top-4 : " + ", ".join(f"{f}={v:.1e}" for f, v in worst))
    print(f"  NaN mismatches   : {nan_mis}")
    print(f"  -> {'PASS' if test_a else 'FAIL: API computes features differently from training'}")
    print("\nTEST B  end-to-end (FAISS + store + features + ranker)")
    print(f"  same candidate set : {same_set}/{n} ({same_set / n:.1%})")
    print(f"  same top-10 order  : {same_top10}/{n} ({same_top10 / n:.1%})")
    print(f"  top-10 mismatches explained by near-ties (<{NEAR_TIE:g}): {explained}/{n - same_top10}")
    if unexplained:
        print(f"  UNEXPLAINED mismatches for uids: {unexplained[:10]}")
    ok = test_a and same_set / n >= 0.99 and not unexplained
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
