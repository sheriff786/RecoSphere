"""
Candidate generation + ranker features — ONE implementation used by:
  notebook 05 (ranker training/eval), notebook 06 (beyond accuracy), and the serving API (Phase 6).
Same code everywhere = no training-serving skew.

Rule: every feature is computed only from `hist_df`, which must end strictly before `t_ref`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch

from eval.metrics import build_matrix, topk
from retrieval.two_tower import embed_items, embed_users, history_at_cutoff

FEATS = ["tt_score", "tt_rank", "pop_rank", "in_tt", "in_pop",
         "i_pop_all", "i_pop_30d", "i_pop_90d", "i_pop_365d", "i_trend", "i_age_days", "i_dfa_missing",
         "store", "type", "rated",
         "u_n_hist", "u_days_since_last", "u_span_days", "u_mean_rating",
         "ui_store_share", "ui_text_sim_mean", "ui_text_sim_last"]
CAT_FEATS = ["store", "type", "rated"]


TIE_MARGIN = 50     # extra candidates fetched so that ties at the cut-off are resolved the same way everywhere


def tie_safe_top(items: np.ndarray, scores: np.ndarray, n: int) -> np.ndarray:
    """Top-n by score, ties broken by item id. FAISS and numpy order equal scores differently; without a
    fixed rule the tt_rank feature (and so the ranker score) would differ between notebook and API."""
    return items[np.lexsort((items, -scores))][:n]


@dataclass
class RankContext:
    n_users: int
    n_items: int
    hist_len: int
    dfa_t: torch.Tensor          # for two-tower item age (if the variant uses it)
    text_emb: np.ndarray
    store: np.ndarray
    type_: np.ndarray
    rated: np.ndarray
    dfa: np.ndarray              # Date First Available, seconds (-1 = missing)
    n_tt: int = 200
    n_pop: int = 100
    recent_frac: float = 0.1


def make_context(item_feat: pl.DataFrame, text_emb, dfa_t, n_users, hist_len, n_tt=200, n_pop=100, recent_frac=0.1):
    item_feat = item_feat.sort("iid")
    return RankContext(
        n_users=n_users, n_items=item_feat.height, hist_len=hist_len, dfa_t=dfa_t, text_emb=text_emb,
        store=item_feat["store_id"].to_numpy(), type_=item_feat["type_id"].to_numpy(),
        rated=item_feat["rated_id"].to_numpy(),
        dfa=item_feat["dfa_ts"].fill_null(-1).to_numpy().astype(np.float64),
        n_tt=n_tt, n_pop=n_pop, recent_frac=recent_frac)


def recent_popularity(ctx: RankContext, hist_df: pl.DataFrame) -> np.ndarray:
    """Item counts in the last `recent_frac` of history interactions (same as the baseline)."""
    t_rec = hist_df["ts"].quantile(1 - ctx.recent_frac, interpolation="lower")
    return np.bincount(hist_df.filter(pl.col("ts") >= t_rec)["iid"].to_numpy(), minlength=ctx.n_items)


def gen_candidates(ctx: RankContext, model, hist_df: pl.DataFrame, win_df: pl.DataFrame, t_ref: int,
                   V: np.ndarray | None = None) -> dict:
    """Union of two-tower top-n_tt and recent-popularity top-n_pop (unseen items) for every user who has
    history and at least one NEW item in the window. Rows are grouped by user in ascending uid order."""
    N = ctx.n_items
    X = build_matrix(hist_df, ctx.n_users, N)
    has_hist = np.diff(X.indptr) > 0
    pairs = win_df.select("uid", "iid").unique()
    tu, ti = pairs["uid"].to_numpy(), pairs["iid"].to_numpy()
    keep = has_hist[tu] & (np.asarray(X[tu, ti]).ravel() == 0)          # only NEW items are targets
    tu, ti = tu[keep], ti[keep]
    users = np.unique(tu)

    H = history_at_cutoff(hist_df, ctx.n_users, ctx.hist_len)
    U = embed_users(model, H[users])
    if V is None:                                                         # serving passes its saved item_emb.npy
        V = embed_items(model, ctx.dfa_t, t_ref)
    pop_order = np.argsort(-recent_popularity(ctx, hist_df), kind="stable")[: ctx.n_pop + 1000]

    parts = {k: [] for k in ("u", "i", "tt_rank", "pop_rank", "tt_score")}
    for s in range(0, len(users), 2048):
        ub = users[s:s + 2048]
        S = U[s:s + 2048] @ V.T
        for r, u in enumerate(ub):
            S[r, X.indices[X.indptr[u]:X.indptr[u + 1]]] = -np.inf             # seen items mask
        top = topk(S, min(N, ctx.n_tt + TIE_MARGIN))
        for r, u in enumerate(ub):
            seen = X.indices[X.indptr[u]:X.indptr[u + 1]]
            cand = top[r][np.isfinite(S[r, top[r]])]
            tt_arr = tie_safe_top(cand, S[r, cand], ctx.n_tt)
            pop_arr = pop_order[~np.isin(pop_order, seen)][: ctx.n_pop]
            items = np.concatenate([tt_arr, pop_arr[~np.isin(pop_arr, tt_arr)]])   # union, order preserved
            ttr = np.full(len(items), np.nan, np.float32); ttr[:len(tt_arr)] = np.arange(1, len(tt_arr) + 1)
            srt = np.argsort(pop_arr); j = np.searchsorted(pop_arr[srt], items).clip(0, len(pop_arr) - 1)
            ppr = np.where(pop_arr[srt][j] == items, srt[j] + 1, np.nan).astype(np.float32)
            parts["u"].append(np.full(len(items), u, np.int64)); parts["i"].append(items.astype(np.int64))
            parts["tt_rank"].append(ttr); parts["pop_rank"].append(ppr)
            parts["tt_score"].append(S[r, items].astype(np.float32))
    c = {k: np.concatenate(v) for k, v in parts.items()}
    c.update(users=users, X=X, H=H, t_ref=int(t_ref), target_u=tu, target_i=ti)
    tkeys = np.sort(tu * N + ti)
    c["label"] = np.isin(c["u"] * N + c["i"], tkeys).astype(np.int8)
    n_t = np.bincount(tu, minlength=ctx.n_users)[users]

    def cand_recall(mask):
        return float(np.mean(np.bincount(c["u"][mask & (c["label"] == 1)], minlength=ctx.n_users)[users] / n_t))
    c["recall"] = {"two_tower": cand_recall(~np.isnan(c["tt_rank"])), "recent_pop": cand_recall(~np.isnan(c["pop_rank"])),
                   "union": cand_recall(np.ones(len(c["u"]), bool))}
    return c


def _rowwise_dot(A_idx, A, B_idx, B, chunk=200_000):
    out = np.empty(len(A_idx), dtype=np.float32)
    for s in range(0, len(A_idx), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", A[A_idx[s:s + chunk]], B[B_idx[s:s + chunk]])
    return out


# ---------------------------------------------------------------------------
# Features = (item state) + (user block) -> assemble().
# Batch (notebooks) builds both from a history DataFrame; serving builds the item state once at
# startup and the user block for ONE user from the feature store. assemble() is shared, so the
# feature math cannot drift between training and serving.
# ---------------------------------------------------------------------------
@dataclass
class ItemState:
    t_ref: int
    pop_all: np.ndarray
    pop_30d: np.ndarray
    pop_90d: np.ndarray
    pop_365d: np.ndarray
    age_days: np.ndarray
    dfa_missing: np.ndarray
    is_new: np.ndarray           # never seen in history (cold)
    recent_pop_order: np.ndarray # items by recent popularity (baseline definition), best first


@dataclass
class UserBlock:
    n_hist: np.ndarray           # (k,)
    first: np.ndarray            # (k,) first interaction ts
    last: np.ndarray             # (k,) last interaction ts
    sum_rating: np.ndarray       # (k,)
    H: np.ndarray                # (k, hist_len) last items, most recent in the last slot, PAD = -1
    store_counts: sp.csr_matrix  # (k, n_store) how many history items per publisher


def build_item_state(ctx: RankContext, hist_df: pl.DataFrame, t_ref: int) -> ItemState:
    hi, ht = hist_df["iid"].to_numpy(), hist_df["ts"].to_numpy()
    assert ht.max() < t_ref, "history contains rows at/after the cutoff!"          # LEAKAGE GUARD
    cnt = lambda m: np.bincount(hi[m], minlength=ctx.n_items).astype(np.float32)
    pop_all = cnt(np.ones(len(hi), bool))
    age = np.clip((t_ref - ctx.dfa) / 86400, 0, None).astype(np.float32); age[ctx.dfa < 0] = np.nan
    return ItemState(t_ref=int(t_ref), pop_all=pop_all,
                     pop_30d=cnt(ht >= t_ref - 30 * 86400), pop_90d=cnt(ht >= t_ref - 90 * 86400),
                     pop_365d=cnt(ht >= t_ref - 365 * 86400), age_days=age,
                     dfa_missing=(ctx.dfa < 0).astype(np.float32), is_new=pop_all == 0,
                     recent_pop_order=np.argsort(-recent_popularity(ctx, hist_df), kind="stable"))


def build_user_block(ctx: RankContext, hist_df: pl.DataFrame, users: np.ndarray, H_all: np.ndarray) -> UserBlock:
    hu, hi = hist_df["uid"].to_numpy(), hist_df["iid"].to_numpy()
    ht, hr = hist_df["ts"].to_numpy(), hist_df["rating"].to_numpy().astype(np.float64)
    NU = max(ctx.n_users, int(hu.max()) + 1)
    n_hist = np.bincount(hu, minlength=NU).astype(np.float32)
    last = np.full(NU, -np.inf); np.maximum.at(last, hu, ht)
    first = np.full(NU, np.inf); np.minimum.at(first, hu, ht)
    sum_r = np.bincount(hu, weights=hr, minlength=NU)
    M = sp.csr_matrix((np.ones(len(hu), np.float32), (hu, ctx.store[hi])), shape=(NU, int(ctx.store.max()) + 1))
    return UserBlock(n_hist=n_hist[users], first=first[users], last=last[users], sum_rating=sum_r[users],
                     H=H_all[users], store_counts=M[users])


def assemble(ctx: RankContext, items: ItemState, ub: UserBlock, row_user: np.ndarray, i: np.ndarray,
             tt_score: np.ndarray, tt_rank: np.ndarray, pop_rank: np.ndarray) -> pd.DataFrame:
    """row_user = index into the user block for every candidate row."""
    t_ref, E = items.t_ref, ctx.text_emb
    valid = ub.H >= 0                                                       # user text profile
    prof = np.zeros((len(ub.H), E.shape[1]), np.float32)
    for s in range(0, len(ub.H), 4096):
        prof[s:s + 4096] = (E[np.clip(ub.H[s:s + 4096], 0, None)] * valid[s:s + 4096, :, None]).sum(1)
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8
    last_emb = E[np.clip(ub.H[:, -1], 0, None)]

    st = ctx.store[i]
    n_hist = ub.n_hist[row_user]
    mean_r = (ub.sum_rating / np.maximum(ub.n_hist, 1)).astype(np.float32)
    F = {
        "tt_score": tt_score, "tt_rank": tt_rank, "pop_rank": pop_rank,
        "in_tt": (~np.isnan(tt_rank)).astype(np.float32), "in_pop": (~np.isnan(pop_rank)).astype(np.float32),
        "i_pop_all": items.pop_all[i], "i_pop_30d": items.pop_30d[i], "i_pop_90d": items.pop_90d[i],
        "i_pop_365d": items.pop_365d[i], "i_trend": items.pop_30d[i] / (items.pop_365d[i] + 1),
        "i_age_days": items.age_days[i], "i_dfa_missing": items.dfa_missing[i],
        "store": st.astype(np.float32), "type": ctx.type_[i].astype(np.float32), "rated": ctx.rated[i].astype(np.float32),
        "u_n_hist": n_hist, "u_days_since_last": ((t_ref - ub.last[row_user]) / 86400).astype(np.float32),
        "u_span_days": ((ub.last[row_user] - ub.first[row_user]) / 86400).astype(np.float32),
        "u_mean_rating": mean_r[row_user],
        "ui_store_share": np.where(st > 0, np.asarray(ub.store_counts[row_user, st]).ravel() / np.maximum(n_hist, 1),
                                   np.nan).astype(np.float32),
        "ui_text_sim_mean": _rowwise_dot(i, E, row_user, prof),
        "ui_text_sim_last": _rowwise_dot(i, E, row_user, last_emb),
    }
    return pd.DataFrame({k: F[k] for k in FEATS})


def features(ctx: RankContext, c: dict, hist_df: pl.DataFrame) -> pd.DataFrame:
    """Batch path (notebooks): state from hist_df, then the shared assemble()."""
    items = build_item_state(ctx, hist_df, c["t_ref"])
    ub = build_user_block(ctx, hist_df, c["users"], c["H"])
    return assemble(ctx, items, ub, np.searchsorted(c["users"], c["u"]), c["i"],
                    c["tt_score"], c["tt_rank"], c["pop_rank"])


def new_item_slots(top: np.ndarray, is_new: np.ndarray, k: int = 10, min_new: int = 2) -> np.ndarray:
    """Re-rank chosen in notebook 06: take top-k, then make sure >= min_new never-seen items are in it
    (replacing the lowest-ranked old items). `top` = ranked item ids, best first."""
    chosen = [int(x) for x in top[:k]]
    need = min_new - int(sum(is_new[x] for x in chosen))
    extra = [int(x) for x in top[k:] if is_new[x]][:max(need, 0)]
    for e in extra:
        old = [j for j in range(len(chosen) - 1, -1, -1) if not is_new[chosen[j]]]
        if not old:
            break
        chosen[old[0]] = e
    pos = {int(x): j for j, x in enumerate(top)}
    chosen.sort(key=pos.get)                       # final list keeps the ranker's order
    return np.array(chosen, dtype=np.int64)
