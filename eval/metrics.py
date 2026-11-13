"""
Offline eval module — har model (baseline, two-tower, ranker) isi se evaluate hoga.
Ek hi metric code = numbers comparable. Ye project ka sabse important file hai.

Protocol:
  - Eval users = jo history (train) mein bhi hain. Pure cold users alag problem hai.
  - Target = user ke eval-window items jo history mein NAHI the (re-purchase predict nahi kar rahe).
  - History items score se mask hote hain (already-seen recommend karna cheating hai).
  - Slices: 'all' targets, 'warm' (item train mein tha), 'cold' (naya item).
    ID-based models (EASE, item-CF) cold items kabhi retrieve nahi kar sakte -> cold recall = 0.
    Yahi gap text-aware item tower (Phase 2) bharega.

Metrics:
  Recall@K = |topK ∩ targets| / |targets|
  NDCG@K   = DCG / ideal DCG, binary relevance, log2 discount
"""
from collections import defaultdict
from typing import Callable

import numpy as np
import polars as pl
import scipy.sparse as sp


def build_matrix(df: pl.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(df.height, dtype=np.float32), (df["uid"].to_numpy(), df["iid"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.data[:] = 1.0  # duplicate (u,i) -> binary
    return m


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k indices, sorted by score desc. argpartition = O(n), full sort nahi."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    return part[row, np.argsort(-scores[row, part], axis=1)]


def _ndcg(hits: np.ndarray, n_targets: int, k: int) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc[: len(hits)]).sum()
    idcg = disc[: min(n_targets, k)].sum()
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    score_fn: Callable[[np.ndarray], np.ndarray],
    history: sp.csr_matrix,
    eval_df: pl.DataFrame,
    known_items: np.ndarray,
    ks=(10, 50, 100),
    batch_size: int = 1024,
    return_per_user: bool = False,
) -> dict:
    """
    score_fn(user_ids) -> dense scores [len(user_ids), n_items]
    history         -> CSR of interactions the model was allowed to see (masking ke liye)
    eval_df         -> uid, iid of the eval window
    known_items     -> items present in history (warm/cold slice ke liye)
    return_per_user -> out["per_user"] = {"uid": [...], "all/ndcg@10": [...], ...} ('all' slice; for the
                       paired bootstrap in the Phase 7 gate — same users in the same order for every model)
    """
    n_users = history.shape[0]
    hist_users = np.flatnonzero(np.diff(history.indptr) > 0)
    known = np.zeros(history.shape[1], dtype=bool)
    known[known_items] = True

    # Vectorized: eval pairs jo history mein already hain, unhe hatao
    pairs = eval_df.select("uid", "iid").unique()
    u_arr, i_arr = pairs["uid"].to_numpy(), pairs["iid"].to_numpy()
    keep = u_arr < n_users
    u_arr, i_arr = u_arr[keep], i_arr[keep]
    seen = np.asarray(history[u_arr, i_arr]).ravel() > 0
    targets = defaultdict(set)
    for u, i in zip(u_arr[~seen].tolist(), i_arr[~seen].tolist()):
        targets[u].add(i)
    hist_set = set(hist_users.tolist())
    users = np.array(sorted(u for u in targets if u in hist_set), dtype=np.int64)

    kmax = max(ks)
    acc = {s: {f"{m}@{k}": [] for m in ("recall", "ndcg") for k in ks} for s in ("all", "warm", "cold")}
    cold_share = []

    for start in range(0, len(users), batch_size):
        ub = users[start:start + batch_size]
        scores = np.asarray(score_fn(ub), dtype=np.float32)
        h = history[ub]
        scores[h.nonzero()] = -np.inf  # already-seen mask
        top = topk(scores, kmax)
        for row, u in enumerate(ub):
            t_all = targets[u]
            slices = {
                "all": t_all,
                "warm": {i for i in t_all if known[i]},
                "cold": {i for i in t_all if not known[i]},
            }
            cold_share.append(len(slices["cold"]) / len(t_all))
            for s, t in slices.items():
                if not t:
                    continue
                hits_full = np.fromiter((i in t for i in top[row]), dtype=np.float32, count=len(top[row]))
                for k in ks:
                    h_k = hits_full[:k]
                    acc[s][f"recall@{k}"].append(h_k.sum() / len(t))
                    acc[s][f"ndcg@{k}"].append(_ndcg(h_k, len(t), k))

    per_user = {"uid": users} | {f"all/{m}": np.asarray(v, dtype=np.float32) for m, v in acc["all"].items()}
    out = {"n_eval_users": len(users), "mean_cold_target_share": float(np.mean(cold_share)) if cold_share else 0.0}
    for s, d in acc.items():
        out[f"n_users_{s}"] = len(d[f"recall@{ks[0]}"])
        for m, v in d.items():
            out[f"{s}/{m}"] = float(np.mean(v)) if v else 0.0
    if return_per_user:
        out["per_user"] = per_user
    return out
