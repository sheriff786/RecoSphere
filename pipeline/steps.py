"""
Training / evaluation steps. Each step reads and writes files in the run directory only, so any step
can be re-run on its own:  python -m pipeline.run --run-id <id> --steps train_ranker,evaluate

Run directory = everything the serving API needs (+ reports):
  two_tower.pt  two_tower_trainonly.pt  item_emb.npy  faiss.index  faiss_meta.json
  ranker.txt  ranker_meta.json  serving_policy.json  metrics.json  data/
"""
import hashlib
import json
import time
import warnings

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import torch

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

from eval.metrics import build_matrix, evaluate
from pipeline.config import VARIANTS, two_tower_cfg
from ranking.pipeline import (CAT_FEATS, FEATS, features, gen_candidates, make_context, new_item_slots,
                              recent_popularity)
from retrieval.two_tower import embed_items, load_two_tower, save_checkpoint, train_two_tower

SHOW = ["all/recall@50", "all/recall@100", "all/ndcg@10", "warm/recall@100", "cold/recall@100"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _dfa(rd):
    return torch.tensor(rd.item_feat["dfa_ts"].fill_null(-1).to_numpy(), dtype=torch.float64, device=DEVICE)


# ---------------------------------------------------------------------------------- step 2
def train_retrieval(cfg, rd, log=print) -> dict:
    """Two models, same recipe: train-only (makes the ranker's VAL candidates without leakage)
    and train+val (production retrieval, evaluated on TEST)."""
    tcfg = two_tower_cfg(cfg)
    variant = cfg["two_tower"]["variant"]; flags = VARIANTS[variant]
    text_emb, dfa = rd.text_emb(tcfg["text_encoder"]), _dfa(rd)
    out = {}
    for name, fit_df, t_ref, fname in (("train-only", rd.train, rd.t_val, "two_tower_trainonly.pt"),
                                       ("train+val", rd.train_val, rd.t_test, "two_tower.pt")):
        t0 = time.time()
        log(f"two-tower [{name}] {variant}, {tcfg['epochs']} epochs on {fit_df.height:,} rows ({DEVICE})")
        m = train_two_tower(flags, tcfg, fit_df, tcfg["epochs"], rd.item_feat, text_emb, dfa, DEVICE,
                            log=lambda s: log("  " + s.strip()))
        save_checkpoint(m.cpu(), rd.run_dir / fname, variant, flags, tcfg, rd.n_users, t_ref)
        out[f"time/two_tower_{name}_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------------- step 3
def build_index(cfg, rd, log=print) -> dict:
    model, ck, dfa = load_two_tower(rd.run_dir / "two_tower.pt", rd.gold, "cpu")
    V = embed_items(model, dfa, ck["t_ref"])
    np.save(rd.run_dir / "item_emb.npy", V)
    ic, d = cfg["index"], V.shape[1]
    nlist = max(16, int(2 * np.sqrt(len(V))))
    if ic["type"] == "flat":
        idx = faiss.IndexFlatIP(d)
    elif ic["type"] == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = 200
    elif ic["type"] == "ivf":
        idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT); idx.train(V)
    elif ic["type"] == "ivfpq":
        idx = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 16, 8, faiss.METRIC_INNER_PRODUCT); idx.train(V)
    else:
        raise ValueError(ic["type"])
    idx.add(V)
    faiss.write_index(idx, str(rd.run_dir / "faiss.index"))
    ver = hashlib.sha1(V.tobytes()).hexdigest()[:12]
    (rd.run_dir / "faiss_meta.json").write_text(json.dumps(
        {"index": ic["type"], "knob": ic["knob"], "k_search": ic["k_search"], "emb_version": ver,
         "n_items": len(V), "dim": d, "nlist": nlist, "model_variant": ck["variant"]}, indent=2))
    log(f"index: {ic['type']} over {len(V):,} items | emb_version {ver}")
    return {}


# ---------------------------------------------------------------------------------- step 4
def train_ranker(cfg, rd, log=print) -> dict:
    rc = cfg["ranker"]
    model_tr, ck, dfa = load_two_tower(rd.run_dir / "two_tower_trainonly.pt", rd.gold, DEVICE)
    ctx = make_context(rd.item_feat, rd.text_emb(ck["cfg"]["text_encoder"]), dfa, rd.n_users,
                       ck["cfg"]["hist_len"], rc["n_tt"], rc["n_pop"], rc["recent_frac"])
    t0 = time.time()
    c = gen_candidates(ctx, model_tr, rd.train, rd.val, rd.t_val)
    F = features(ctx, c, rd.train)
    log(f"ranker data (val window): {len(c['users']):,} users, {len(c['u']):,} rows, "
        f"candidate recall {c['recall']['union']:.4f} ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(rc["seed"])
    pos_users = np.unique(c["u"][c["label"] == 1])
    chosen = rng.permutation(pos_users)[: rc["max_train_users"]]
    groups = lambda u: np.unique(u, return_counts=True)[1]
    params = dict(objective="lambdarank", learning_rate=rc["learning_rate"], num_leaves=rc["num_leaves"],
                  min_child_samples=rc["min_child_samples"], colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                  random_state=rc["seed"], verbose=-1)
    k = int(rc.get("cv_folds", 0))
    t0 = time.time()
    extra = {}
    if k >= 2:
        # Number of trees by K-fold CV over USERS: average the K validation NDCG@10 curves and take the
        # argmax of the MEAN curve. One ~15% split is ~800 users on real data -> its curve is so noisy that
        # early stopping often locks onto a random early peak (we saw best_iteration 14 and 18). The mean
        # curve uses every user once; the final model is then trained on ALL users with that many trees.
        folds = np.array_split(chosen, k)
        curves, fold_best = [], []
        for f in range(k):
            es_u, fit_u = np.sort(folds[f]), np.sort(np.concatenate([folds[j] for j in range(k) if j != f]))
            m_fit, m_es = np.isin(c["u"], fit_u), np.isin(c["u"], es_u)
            r = lgb.LGBMRanker(n_estimators=rc["n_estimators"], **params)
            r.fit(F[m_fit], c["label"][m_fit], group=groups(c["u"][m_fit]),
                  eval_set=[(F[m_es], c["label"][m_es])], eval_group=[groups(c["u"][m_es])], eval_at=[10],
                  categorical_feature=CAT_FEATS, callbacks=[lgb.early_stopping(rc.get("cv_patience", 100), verbose=False)])
            curves.append(np.asarray(r.evals_result_["valid_0"]["ndcg@10"]))
            fold_best.append(int(np.argmax(curves[-1]) + 1))
        L = min(len(x) for x in curves)
        mean = np.mean([x[:L] for x in curves], axis=0)
        best = int(np.argmax(mean) + 1)
        log(f"ranker CV ({k} folds, {len(chosen):,} users, ~{len(chosen) // k:,} per fold): best per fold {fold_best} "
            f"-> mean curve best {best} (cv ndcg@10 {mean[best - 1]:.4f})")
        ranker = lgb.LGBMRanker(n_estimators=best, **params)
        m_all = np.isin(c["u"], np.sort(chosen))
        ranker.fit(F[m_all], c["label"][m_all], group=groups(c["u"][m_all]), categorical_feature=CAT_FEATS)
        extra = {"ranker/cv_ndcg10": float(mean[best - 1]), "ranker/cv_fold_best_std": float(np.std(fold_best))}
    else:                                   # legacy: one 15% early-stopping split (notebook 05)
        n_es = max(1, int(0.15 * len(chosen)))
        es_u, fit_u = np.sort(chosen[:n_es]), np.sort(chosen[n_es:])
        m_fit, m_es = np.isin(c["u"], fit_u), np.isin(c["u"], es_u)
        ranker = lgb.LGBMRanker(n_estimators=rc["n_estimators"], **params)
        ranker.fit(F[m_fit], c["label"][m_fit], group=groups(c["u"][m_fit]),
                   eval_set=[(F[m_es], c["label"][m_es])], eval_group=[groups(c["u"][m_es])], eval_at=[10],
                   categorical_feature=CAT_FEATS, callbacks=[lgb.early_stopping(rc["early_stopping"], verbose=False)])
        log(f"ranker: {len(chosen):,} users with a positive candidate, early stopping on {n_es:,}")
    if k < 2:
        best = int(ranker.best_iteration_ or rc["n_estimators"])
    ranker.booster_.save_model(str(rd.run_dir / "ranker.txt"), num_iteration=best)
    (rd.run_dir / "ranker_meta.json").write_text(json.dumps(
        {"features": FEATS, "categorical": CAT_FEATS, "n_tt": rc["n_tt"], "n_pop": rc["n_pop"],
         "recent_frac": rc["recent_frac"], "best_iteration": best, "retrieval_variant": ck["variant"],
         "params": {k: v for k, v in ranker.get_params().items() if isinstance(v, (int, float, str, type(None)))}},
        indent=2))
    log(f"ranker: best iteration {best} ({time.time() - t0:.0f}s)")
    return {"ranker/best_iteration": best, "ranker/val_candidate_recall": c["recall"]["union"],
            "ranker/n_train_users": len(chosen), **extra}


# ---------------------------------------------------------------------------------- step 5
def choose_fallback(rd) -> tuple[str, dict]:
    """Notebook 06: non-personal list for users without history — chosen on VAL (history = train)."""
    X = build_matrix(rd.train, rd.n_users, rd.n_items)
    no_hist = np.diff(X.indptr) == 0
    w = rd.val.filter(pl.col("uid").is_in(np.flatnonzero(no_hist).tolist())).unique(["uid", "iid"])
    if w.height == 0:
        return "recent_popularity_10pct", {}
    users = np.unique(w["uid"].to_numpy())
    tk = np.sort(w["uid"].to_numpy() * rd.n_items + w["iid"].to_numpy())
    n_t = np.bincount(w["uid"].to_numpy(), minlength=rd.n_users)[users]
    hi, ht = rd.train["iid"].to_numpy(), rd.train["ts"].to_numpy()
    t_ref = rd.t_val
    t_rec = rd.train["ts"].quantile(0.9, interpolation="lower")
    opts = {"all_time_popularity": np.bincount(hi, minlength=rd.n_items),
            "recent_popularity_10pct": np.bincount(hi[ht >= t_rec], minlength=rd.n_items),
            "last_30_days": np.bincount(hi[ht >= t_ref - 30 * 86400], minlength=rd.n_items),
            "last_90_days": np.bincount(hi[ht >= t_ref - 90 * 86400], minlength=rd.n_items)}
    scores = {}
    for name, cnt in opts.items():
        top = np.argsort(-cnt, kind="stable")[:100]
        hits = np.isin(users[:, None] * rd.n_items + top[None, :], tk)
        scores[name] = float((hits.sum(1) / n_t).mean())
    return max(scores, key=scores.get), scores


def evaluate_models(model_dir, rd, new_item_slots_k: int = 2, log=print, per_user_path=None) -> dict:
    """Evaluate the models in `model_dir` on the TEST window of run data `rd`.
    The gate calls this for the production models too, so both are measured on the same data.
    per_user_path -> also save per-user ranker NDCG/recall + candidate recall (.npz) for the paired bootstrap."""
    model, ck, dfa = load_two_tower(model_dir / "two_tower.pt", rd.gold, DEVICE)
    rmeta = json.loads((model_dir / "ranker_meta.json").read_text())
    booster = lgb.Booster(model_file=str(model_dir / "ranker.txt"))
    ctx = make_context(rd.item_feat, rd.text_emb(ck["cfg"]["text_encoder"]), dfa, rd.n_users,
                       ck["cfg"]["hist_len"], rmeta["n_tt"], rmeta["n_pop"], rmeta["recent_frac"])
    V = np.load(model_dir / "item_emb.npy")
    t0 = time.time()
    c = gen_candidates(ctx, model, rd.train_val, rd.test, rd.t_test, V=V)
    F = features(ctx, c, rd.train_val)
    known = rd.train_val["iid"].unique().to_numpy()
    starts = np.searchsorted(c["u"], c["users"]); ends = np.append(starts[1:], len(c["u"]))
    where = dict(zip(c["users"].tolist(), zip(starts.tolist(), ends.tolist())))

    def ev(scores, ks=(10, 50, 100), per_user=False):
        def fn(ub):
            S = np.full((len(ub), rd.n_items), -np.inf, dtype=np.float32)
            for r, u in enumerate(ub):
                if u in where:
                    s0, s1 = where[u]; S[r, c["i"][s0:s1]] = scores[s0:s1]
            return S
        return evaluate(fn, c["X"], rd.test, known, ks=ks, return_per_user=per_user)

    ranker_scores = booster.predict(F).astype(np.float32)
    orders = {"ranker": ranker_scores, "two_tower": c["tt_score"],
              "recent_pop": np.where(np.isnan(c["pop_rank"]), -1e9, -np.nan_to_num(c["pop_rank"])).astype(np.float32)}
    out, pu = {}, {}
    for name, s in orders.items():
        r = ev(s, per_user=(name == "ranker"))
        out.update({f"{name}/{k}": float(r[k]) for k in SHOW})
        if name == "ranker":
            pu = {"uid": r["per_user"]["uid"], "ranker/all/ndcg@10": r["per_user"]["all/ndcg@10"],
                  "ranker/all/recall@100": r["per_user"]["all/recall@100"]}
    out["eval/n_users"] = int(r["n_eval_users"])
    out.update({f"candidates/recall_{k}": float(v) for k, v in c["recall"].items()})
    if per_user_path is not None:
        n_t = np.bincount(c["target_u"], minlength=rd.n_users)
        hit = np.bincount(c["u"][c["label"] == 1], minlength=rd.n_users)
        assert np.array_equal(pu["uid"], c["users"]), "eval users != candidate users"
        pu["candidates/recall_union"] = (hit[c["users"]] / n_t[c["users"]]).astype(np.float32)
        np.savez_compressed(per_user_path, **{k.replace("/", "|"): v for k, v in pu.items()})

    # serving policy: ranker top-100 -> new-item slots -> top-10
    is_new = np.bincount(rd.train_val["iid"].to_numpy(), minlength=rd.n_items) == 0
    rr = np.full(len(ranker_scores), -np.inf, dtype=np.float32)
    for s0, s1 in zip(starts, ends):
        items = c["i"][s0:s1]
        top = items[np.argsort(-ranker_scores[s0:s1], kind="stable")][:100]
        final = new_item_slots(top, is_new, 10, new_item_slots_k)
        pos = {int(x): j for j, x in enumerate(items)}
        for rank, it in enumerate(final):
            rr[s0 + pos[int(it)]] = 10 - rank
    r = ev(rr, ks=(10,))
    out.update({"served/all/recall@10": float(r["all/recall@10"]), "served/all/ndcg@10": float(r["all/ndcg@10"]),
                "served/cold/recall@10": float(r["cold/recall@10"])})
    log(f"evaluate {model_dir.name}: ranker ndcg@10={out['ranker/all/ndcg@10']:.4f} "
        f"recall@100={out['ranker/all/recall@100']:.4f} | two-tower recall@100={out['two_tower/all/recall@100']:.4f} "
        f"| recent-pop recall@100={out['recent_pop/all/recall@100']:.4f} ({time.time() - t0:.0f}s)")
    return out


def evaluate_step(cfg, rd, log=print) -> dict:
    m = evaluate_models(rd.run_dir, rd, cfg["serving"]["new_item_slots"], log,
                        per_user_path=rd.run_dir / "per_user_eval.npz")
    fb, fb_scores = choose_fallback(rd)
    (rd.run_dir / "serving_policy.json").write_text(json.dumps(
        {"cold_user_fallback": fb, "k_show": 10, "new_item_slots": cfg["serving"]["new_item_slots"],
         "fallback_val_recall@100": fb_scores}, indent=2))
    (rd.run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    log(f"serving policy: fallback = {fb} (chosen on val)")
    return m
