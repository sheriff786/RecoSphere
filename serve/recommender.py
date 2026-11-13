"""
Online recommender — the request-time version of notebooks 03–06.

recommend(user_id):
  1. feature store  -> user's history, seen items, stats            (Redis / memory)
  2. no history     -> fallback popularity list (serving_policy.json, chosen in notebook 06)
  3. user tower     -> 64-d vector                                    (retrieval/two_tower.py)
  4. FAISS          -> two-tower top-N_TT unseen  + recent-pop top-N_POP unseen = candidates
  5. features       -> ranking/pipeline.assemble()  (SAME code as training -> no skew)
  6. LightGBM       -> scores, sort
  7. re-rank        -> new-item slots (notebook 06 decision) -> top-k
"""
import hashlib
import json
import logging
import time

import faiss
import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from ranking.pipeline import (FEATS, TIE_MARGIN, UserBlock, assemble, build_item_state, make_context,
                              new_item_slots, recent_popularity, tie_safe_top)
from retrieval.two_tower import PAD, embed_users, load_two_tower
from serve.config import ServeConfig
from serve.feature_store import InMemoryStore, RedisStore, bootstrap_records

log = logging.getLogger("recosphere")


class Recommender:
    def __init__(self, cfg: ServeConfig, store=None, model_dir=None):
        """model_dir: serve THIS model folder instead of production.json (A/B treatment arms, Phase 8)."""
        t0 = time.perf_counter()
        torch.set_num_threads(cfg.threads); faiss.omp_set_num_threads(cfg.threads)
        self.cfg = cfg
        G, M = cfg.gold, (model_dir or cfg.models)
        self.model_dir = M                                         # fixed for THIS object's lifetime
        prod = cfg.production
        self.model_run = ("notebook (no production.json)" if M == cfg.models_root
                          else prod["run_id"] if prod and M.name == prod["run_id"] and model_dir is None
                          else M.name)
        self.loaded_at = time.time()

        # --- offline artifacts -------------------------------------------------------------
        train, val = pl.read_parquet(G / "train.parquet"), pl.read_parquet(G / "val.parquet")
        self.hist = pl.concat([train, val])                       # serving "now" = start of test window
        test_path = G / "test.parquet"
        self.t_ref = int(pl.read_parquet(test_path)["ts"].min()) if test_path.exists() else int(self.hist["ts"].max()) + 1
        self.user_map = pl.read_parquet(G / "user_map.parquet")
        item_feat = pl.read_parquet(G / "item_features.parquet").sort("iid")
        self.item_ids = item_feat["item_id"].to_list()
        self.titles = item_feat["title"].fill_null("").to_list()
        self.iid_of = {x: i for i, x in enumerate(self.item_ids)}

        self.model, ck, dfa_t = load_two_tower(M / "two_tower.pt", G, "cpu")
        self.hist_len = ck["cfg"]["hist_len"]
        text_emb = np.load(G / f"item_text_emb_{ck['cfg']['text_encoder']}.npy")
        self.rmeta = json.loads((M / "ranker_meta.json").read_text())
        self.fmeta = json.loads((M / "faiss_meta.json").read_text())
        self.policy = json.loads((M / "serving_policy.json").read_text())
        assert self.rmeta["features"] == FEATS, "ranker trained on different features than ranking/pipeline.FEATS"

        self.ctx = make_context(item_feat, text_emb, dfa_t, self.user_map.height, self.hist_len,
                                self.rmeta["n_tt"], self.rmeta["n_pop"], self.rmeta["recent_frac"])
        self.V = np.load(M / "item_emb.npy")
        self.emb_version = hashlib.sha1(self.V.tobytes()).hexdigest()[:12]
        assert self.emb_version == self.fmeta["emb_version"], \
            f"embedding version mismatch: item_emb {self.emb_version} vs faiss index {self.fmeta['emb_version']}"
        self.index = faiss.read_index(str(M / "faiss.index"))
        if self.fmeta["index"] == "hnsw":
            self.index.hnsw.efSearch = int(self.fmeta["knob"])
        elif self.fmeta["index"] in ("ivf", "ivfpq"):
            self.index.nprobe = int(self.fmeta["knob"])
        self.booster = lgb.Booster(model_file=str(M / "ranker.txt"))

        self.items = build_item_state(self.ctx, self.hist, self.t_ref)  # popularity windows etc. as of t_ref
        self.fallback = self._fallback_order(self.policy["cold_user_fallback"])

        # --- online feature store ------------------------------------------------------------
        # User records depend on the DATA (and hist_len), not on the model -> a model hot-swap with the same
        # data keeps the store as is (live events are not wiped). A new data snapshot rebuilds it.
        self.data_version = f"{self.hist.height}-{int(self.hist['ts'].max())}-h{self.hist_len}"
        self.store = store or (RedisStore(cfg.redis_url, f"rs:{cfg.category}") if cfg.redis_url else InMemoryStore())
        if cfg.force_reload or not self.store.is_loaded(self.data_version):
            if self.store.acquire_load_lock():           # many workers start together: only ONE loads
                try:
                    t1 = time.perf_counter()
                    self.store.load(bootstrap_records(self.hist, self.user_map, self.ctx.store, self.hist_len),
                                    self.user_map.height, self.data_version)
                    self.store.register_users(self.user_map)
                    self.store.mark_loaded(self.data_version)
                    log.info("feature store loaded (%s) in %.1fs", self.store.backend, time.perf_counter() - t1)
                finally:
                    self.store.release_load_lock()
            else:
                log.info("another worker is loading the feature store, waiting ...")
                self.store.wait_until_loaded(self.data_version)
        log.info("recommender ready in %.1fs", time.perf_counter() - t0)

    # -------------------------------------------------------------------------------------------
    def _fallback_order(self, name: str) -> np.ndarray:
        hi, ht = self.hist["iid"].to_numpy(), self.hist["ts"].to_numpy()
        n = self.ctx.n_items
        options = {
            "all_time_popularity": lambda: np.bincount(hi, minlength=n),
            "recent_popularity_10pct": lambda: recent_popularity(self.ctx, self.hist),
            "last_30_days": lambda: np.bincount(hi[ht >= self.t_ref - 30 * 86400], minlength=n),
            "last_90_days": lambda: np.bincount(hi[ht >= self.t_ref - 90 * 86400], minlength=n),
        }
        return np.argsort(-options[name](), kind="stable")

    def smoke_test(self, n_users: int = 20, k: int = 10) -> None:
        """Run real requests through a freshly loaded model BEFORE it takes traffic. Raises on any problem."""
        uids = np.unique(self.hist["uid"].to_numpy())
        pick = uids[np.linspace(0, len(uids) - 1, min(n_users, len(uids))).astype(int)]
        names = dict(zip(self.user_map["uid"].to_list(), self.user_map["user_id"].to_list()))
        for u in pick:
            out = self.recommend(names[int(u)], k=k)
            ids = [x["item_id"] for x in out["items"]]
            assert len(ids) == k and len(set(ids)) == k, f"bad list for {names[int(u)]}: {ids}"
            assert all(i in self.iid_of for i in ids), "unknown item id in output"
            assert all(x["score"] is None or np.isfinite(x["score"]) for x in out["items"]), "non-finite score"
        cold = self.recommend("__smoke_test_unknown_user__", k=k)
        assert len(cold["items"]) == k, "fallback list too short"

    def info(self) -> dict:
        return {"status": "ok", "category": self.cfg.category, "store": self.store.backend,
                "users": self.store.n_users(), "items": self.ctx.n_items, "t_ref": self.t_ref,
                "model_run": self.model_run, "model_dir": str(self.model_dir),
                "model_loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.loaded_at)), "emb_version": self.emb_version, "index": self.fmeta["index"], "index_knob": self.fmeta["knob"],
                "retrieval_variant": self.rmeta["retrieval_variant"], "ranker_iterations": self.rmeta["best_iteration"],
                "cold_user_fallback": self.policy["cold_user_fallback"], "new_item_slots": self.cfg.new_item_slots}

    # -------------------------------------------------------------------------------------------
    def user_block(self, rec: dict) -> UserBlock:
        """One user's feature-store record -> the UserBlock that assemble() expects."""
        H = np.full((1, self.hist_len), PAD, dtype=np.int64)
        h = rec["hist"][-self.hist_len:]
        H[0, self.hist_len - len(h):] = h
        S = int(self.ctx.store.max()) + 1
        st = list(rec["stores"].items())
        return UserBlock(n_hist=np.array([rec["n_hist"]], np.float32), first=np.array([rec["first"]], np.float64),
                         last=np.array([rec["last"]], np.float64), sum_rating=np.array([rec["sum_rating"]], np.float64),
                         H=H, store_counts=sp.csr_matrix((np.array([c for _, c in st], np.float32),
                                                          (np.zeros(len(st), int), np.array([s for s, _ in st], int))),
                                                         shape=(1, S)))

    def score_candidates(self, rec: dict, timings: dict, return_features: bool = False):
        """Candidates + ranker scores for one user. Returns (items, ranker_scores). Used by recommend()
        and by serve/parity_check.py."""
        t = time.perf_counter()
        seen = rec["seen"]
        H = np.full((1, self.hist_len), PAD, dtype=np.int64)
        h = rec["hist"][-self.hist_len:]
        H[0, self.hist_len - len(h):] = h
        u = embed_users(self.model, H)                                   # (1, 64)
        timings["user_tower"] = time.perf_counter() - t

        t = time.perf_counter()
        n_tt, n_pop = self.ctx.n_tt, self.ctx.n_pop
        _, I = self.index.search(u, min(self.ctx.n_items, n_tt + len(seen) + TIE_MARGIN))
        cand = I[0][(I[0] >= 0) & ~np.isin(I[0], seen)]
        tt_arr = tie_safe_top(cand, (self.V[cand] @ u[0]).astype(np.float32), n_tt)
        po = self.items.recent_pop_order[: n_pop + len(seen)]
        pop_arr = po[~np.isin(po, seen)][:n_pop]
        items = np.concatenate([tt_arr, pop_arr[~np.isin(pop_arr, tt_arr)]]).astype(np.int64)
        ttr = np.full(len(items), np.nan, np.float32); ttr[:len(tt_arr)] = np.arange(1, len(tt_arr) + 1)
        srt = np.argsort(pop_arr); j = np.searchsorted(pop_arr[srt], items).clip(0, len(pop_arr) - 1)
        ppr = np.where(pop_arr[srt][j] == items, srt[j] + 1, np.nan).astype(np.float32)
        tt_score = (self.V[items] @ u[0]).astype(np.float32)
        timings["ann_candidates"] = time.perf_counter() - t

        t = time.perf_counter()
        ub = self.user_block(rec)
        X = assemble(self.ctx, self.items, ub, np.zeros(len(items), np.int64), items, tt_score, ttr, ppr)
        timings["features"] = time.perf_counter() - t

        t = time.perf_counter()
        scores = self.booster.predict(X, num_threads=self.cfg.threads).astype(np.float32)
        timings["ranker"] = time.perf_counter() - t
        return (items, scores, X) if return_features else (items, scores)

    def recommend(self, user_id: str, k: int = 10, rerank: bool = True) -> dict:
        T0 = time.perf_counter()
        timings = {}
        t = time.perf_counter()
        uid, rec = self.store.get(user_id)
        timings["store_get"] = time.perf_counter() - t

        if rec is None or rec["n_hist"] == 0:
            top, scores, source = self.fallback[:k], None, "fallback"
        else:
            items, scores_all = self.score_candidates(rec, timings)
            t = time.perf_counter()
            order = np.argsort(-scores_all, kind="stable")
            ranked = items[order]
            top = (new_item_slots(ranked[:100], self.items.is_new, k, self.cfg.new_item_slots)
                   if rerank and self.cfg.new_item_slots > 0 else ranked[:k])
            score_of = dict(zip(items.tolist(), scores_all.tolist()))
            scores = [score_of[int(i)] for i in top]
            timings["rerank"] = time.perf_counter() - t
            source = "personalized"

        timings["total"] = time.perf_counter() - T0
        return {"user_id": user_id, "source": source,
                "items": [{"item_id": self.item_ids[i], "title": self.titles[i], "is_new": bool(self.items.is_new[i]),
                           "score": None if scores is None else round(float(scores[n]), 5)} for n, i in enumerate(top)],
                "timings_ms": {k_: round(v * 1e3, 3) for k_, v in timings.items()}}

    def add_event(self, user_id: str, item_id: str, ts: int | None = None, rating: float | None = None) -> dict:
        iid = self.iid_of.get(item_id)
        if iid is None:
            raise KeyError(item_id)
        ts = int(ts if ts is not None else self.t_ref)
        uid, added = self.store.add_event(user_id, iid, ts, float(rating if rating is not None else 5.0),
                                          int(self.ctx.store[iid]), self.hist_len)
        return {"user_id": user_id, "uid": uid, "item_id": item_id, "recorded": added}
