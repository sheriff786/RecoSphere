"""
Product API on top of the engine (for the RecoSphere app, web/).

Everything here READS the live Recommender (same models, same item state, same user store) so the app
shows exactly what the engine would serve:

  catalog     item cards: title, platform/category chips (parsed from the item text), release year,
              popularity, is_new
  search      token match on titles, ties broken by recent popularity
  similar     nearest items in the two-tower item embedding space (the same vectors FAISS serves)
  explain     why an item is in a user's feed: candidate source + rank, and the top LightGBM feature
              contributions (SHAP values, booster.predict(pred_contrib=True)) mapped to plain language,
              plus the most similar game in the user's history ("because you played ...")
  history     the user's recent items from the feature store (what the user tower actually sees)
  demo users  real dataset users with long histories, for "explore as a real player"

Every list takes an optional item_type (game | accessory | other, data/item_types.py) so the app can show
games only in game surfaces and accessories in "Gear for you".
"""
import re
from datetime import datetime, timezone

import numpy as np

from ranking.pipeline import FEATS, new_item_slots, tie_safe_top
from retrieval.two_tower import PAD, embed_users

# feature -> (reason group, plain-language text). User-only features are the same for every candidate of a
# user, so they never explain why THIS item ranks above another -> excluded from reasons.
REASON = {
    "tt_score": ("taste", "Close match to your taste profile"),
    "tt_rank": ("taste", "Ranked high by your taste model"),
    "in_tt": ("taste", "Found by your taste model"),
    "ui_text_sim_mean": ("similar", "Similar to games you've played"),
    "ui_text_sim_last": ("similar", "Similar to the last game you played"),
    "pop_rank": ("trending", "Trending right now"),
    "in_pop": ("trending", "Trending right now"),
    "i_pop_30d": ("trending", "Popular this month"),
    "i_pop_90d": ("trending", "Popular this quarter"),
    "i_trend": ("trending", "Rising fast"),
    "i_pop_365d": ("popular", "Popular this year"),
    "i_pop_all": ("popular", "All-time favourite"),
    "i_age_days": ("fresh", "Recent release"),
    "i_dfa_missing": ("fresh", "Release date unknown"),
    "store": ("publisher", "From a publisher you like"),
    "ui_store_share": ("publisher", "From a publisher you play a lot"),
    "type": ("format", "Format you usually pick"),
    "rated": ("format", "Rating fits your history"),
}
USER_ONLY = {"u_n_hist", "u_days_since_last", "u_span_days", "u_mean_rating"}
# Strength of a reason for users (raw SHAP stays for /admin): its share of everything that pushed THIS item up.
# Thresholds are a product choice (P0.5), not a measured quantity.
STRONG, MEDIUM = 0.40, 0.20


def strength(share: float) -> str:
    return "strong" if share >= STRONG else "medium" if share >= MEDIUM else "weak"
PLATFORMS = [("PlayStation 5", "PS5"), ("PlayStation 4", "PS4"), ("PlayStation 3", "PS3"), ("PS Vita", "Vita"),
             ("Xbox Series", "Xbox Series"), ("Xbox One", "Xbox One"), ("Xbox 360", "Xbox 360"),
             ("Nintendo Switch", "Switch"), ("Wii U", "Wii U"), ("3DS", "3DS"), ("PC", "PC"), ("Mac", "Mac")]


class Product:
    def __init__(self, rec):
        self.rec = rec
        self.n = rec.ctx.n_items
        self.title_tokens = [set(re.findall(r"[a-z0-9]+", (t or "").lower())) for t in rec.titles]
        Vn = rec.V / (np.linalg.norm(rec.V, axis=1, keepdims=True) + 1e-9)
        self.Vn = Vn.astype(np.float32)
        self.pop = rec.items.pop_90d.astype(np.float64)
        self.rank_of_recent = np.empty(self.n, np.int64)
        self.rank_of_recent[rec.items.recent_pop_order] = np.arange(self.n)

    # ------------------------------------------------------------------ catalog
    def _meta(self, i: int) -> dict:
        text = self.rec.texts[i]
        parts = [p.strip() for p in text.split("|")]
        cat = parts[1] if len(parts) > 1 else ""
        crumbs = [c.strip() for c in cat.split(">") if c.strip() and c.strip() != "Video Games"]
        hay = f"{self.rec.titles[i]} {cat}"
        platforms = [short for long_, short in PLATFORMS if long_ in hay][:2]
        desc = parts[-1] if len(parts) > 3 else ""
        dfa = int(self.rec.dfa_ts[i])
        year = datetime.fromtimestamp(int(dfa), tz=timezone.utc).year if dfa and dfa > 0 else None
        return {"item_id": self.rec.item_ids[i], "title": self.rec.titles[i], "platforms": platforms,
                "categories": crumbs[:3], "year": year, "description": desc[:600],
                "is_new": bool(self.rec.items.is_new[i]), "item_type": str(self.rec.item_type[i]), "plays_90d": int(self.rec.items.pop_90d[i]),
                "trending_rank": int(self.rank_of_recent[i]) + 1}

    def items(self, item_ids):
        out = []
        for x in item_ids:
            i = self.rec.iid_of.get(x)
            if i is not None:
                out.append(self._meta(i))
        return out

    def popular(self, k=24, offset=0, new_only=False, item_type=None):
        order = self.rec.keep(self.rec.items.recent_pop_order, item_type)
        if new_only:
            order = order[self.rec.items.is_new[order]]
        return [self._meta(int(i)) for i in order[offset: offset + k]]

    # ------------------------------------------------------------------ fresh
    def fresh(self, user_id: str | None, k=20, item_type=None):
        """Items new to the catalog (no interaction before the cutoff: is_new), the user's seen items removed.
        With history: by taste = user-tower vector . item vector (the model's tt_score; read-only, no ranker:
        its strongest features are popularity windows, all 0 for new items). Without history: newest first
        (first-available date), because popularity cannot order items nobody has played yet."""
        new = self.rec.keep(np.flatnonzero(self.rec.items.is_new), item_type)
        r = self.rec.store.get(user_id)[1] if user_id else None
        if r is not None and r["n_hist"] > 0:
            new = new[~np.isin(new, r["seen"])]
            H = np.full((1, self.rec.hist_len), PAD, dtype=np.int64)
            h = r["hist"][-self.rec.hist_len:]
            H[0, self.rec.hist_len - len(h):] = h
            u = embed_users(self.rec.model, H)[0]
            taste = (self.rec.V[new] @ u).astype(np.float32)
            top = tie_safe_top(new, taste, k)
            score = dict(zip(new.tolist(), taste.tolist()))
            return "taste", [{**self._meta(int(i)), "score": round(float(score[int(i)]), 5)} for i in top]
        dfa = self.rec.dfa_ts[new].astype(np.float64)
        top = new[np.lexsort((new, -np.where(dfa > 0, dfa, -np.inf)))][:k]     # unknown date last, ties by id
        return "newest", [self._meta(int(i)) for i in top]

    # ------------------------------------------------------------------ search
    def search(self, q: str, k=24, item_type=None):
        toks = re.findall(r"[a-z0-9]+", q.lower())
        if not toks:
            return []
        scored = []
        ql = q.lower().strip()
        for i, tt in enumerate(self.title_tokens):
            if item_type is not None and self.rec.item_type[i] != item_type:
                continue
            hit = sum(1 for t in toks if t in tt or any(w.startswith(t) for w in tt))
            if hit:
                exact = 1 if ql in (self.rec.titles[i] or "").lower() else 0
                scored.append((-(hit + exact), self.rank_of_recent[i], i))
        scored.sort()
        return [self._meta(i) for _, _, i in scored[:k]]

    # ------------------------------------------------------------------ similar
    def similar(self, item_id: str, k=12, exclude=(), item_type=None):
        i = self.rec.iid_of[item_id]
        sims = self.Vn @ self.Vn[i]
        sims[i] = -np.inf
        if item_type is not None:
            sims[self.rec.item_type != item_type] = -np.inf
        if exclude:
            ex = [self.rec.iid_of[x] for x in exclude if x in self.rec.iid_of]
            sims[ex] = -np.inf
        k = min(k, int(np.isfinite(sims).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [{**self._meta(int(j)), "similarity": round(float(sims[j]), 3)} for j in top]

    # ------------------------------------------------------------------ user
    def history(self, user_id: str, k=20, item_type=None):
        _, r = self.rec.store.get(user_id)
        if r is None:
            return []
        newest_first = self.rec.keep(np.array(r["hist"][::-1], dtype=np.int64), item_type)
        return [self._meta(int(i)) for i in newest_first[:k]]

    def demo_users(self, k=12, min_hist=8):
        h = self.rec.hist
        import polars as pl
        stats = (h.group_by("uid").agg(pl.len().alias("n"), pl.col("ts").max().alias("last"))
                 .filter(pl.col("n") >= min_hist).sort(["last", "n"], descending=True).head(400))
        if stats.height == 0:
            return []
        # fixed shuffle, then a prefix: demo_users(9) is always a subset of demo_users(50)
        pick = stats.sample(fraction=1.0, shuffle=True, seed=7).head(k).sort("n", descending=True)
        names = dict(zip(self.rec.user_map["uid"].to_list(), self.rec.user_map["user_id"].to_list()))
        out = []
        for uid, n in zip(pick["uid"].to_list(), pick["n"].to_list()):
            last = h.filter(pl.col("uid") == uid).sort("ts")["iid"].to_list()[-3:]
            out.append({"user_id": names[uid], "n_games": n, "recent": [self.rec.titles[i] for i in reversed(last)]})
        return out

    # ------------------------------------------------------------------ explain
    def feed_top10(self, items: np.ndarray, scores: np.ndarray, item_type=None) -> np.ndarray:
        """The user's served top-10 for these candidates: ranker order -> type filter -> new-item slots
        (same steps and arm policy as Recommender.recommend)."""
        ranked = self.rec.keep(items[np.argsort(-scores, kind="stable")], item_type)
        slots = self.rec.cfg.new_item_slots
        return new_item_slots(ranked[:100], self.rec.items.is_new, 10, slots) if slots > 0 else ranked[:10]

    def explain(self, user_id: str, item_ids: list[str] | None = None, top_reasons=3, item_type=None):
        """Explanations for `item_ids` (default: the user's served top-10 for `item_type`). Each item says whether
        it is in that top-10 (`in_top10`); each reason carries its raw SHAP `weight`, its `share` of the item's
        positive contributions and a `strength` label."""
        _, r = self.rec.store.get(user_id)
        if r is None or r["n_hist"] == 0:
            return {"source": "fallback", "items": {x: {"reasons": [{"group": "popular",
                    "text": "Popular with players right now (we don't know your taste yet)"}]} for x in (item_ids or [])}}
        timings = {}
        items, scores, X = self.rec.score_candidates(r, timings, return_features=True)
        order = np.argsort(-scores, kind="stable")
        pos = {int(items[j]): n + 1 for n, j in enumerate(order)}
        idx_of = {int(it): j for j, it in enumerate(items)}
        top10 = {int(i) for i in self.feed_top10(items, scores, item_type)}
        if item_ids is None:
            item_ids = [self.rec.item_ids[int(i)] for i in self.feed_top10(items, scores, item_type)]
        want = [self.rec.iid_of[x] for x in item_ids if x in self.rec.iid_of]
        rows = [idx_of[i] for i in want if i in idx_of]
        Xr = X.iloc[rows] if hasattr(X, "iloc") else X[rows]
        Xv = X.to_numpy(dtype=np.float64) if hasattr(X, "to_numpy") else np.asarray(X, np.float64)
        contrib = self.rec.booster.predict(Xr, pred_contrib=True) if rows else np.zeros((0, len(FEATS) + 1))
        hist = [int(h) for h in r["hist"]]
        out = {}
        for i in want:
            key = self.rec.item_ids[i]
            sim_hist = self.Vn[hist] @ self.Vn[i] if hist else np.array([])
            because = self.rec.titles[hist[int(np.argmax(sim_hist))]] if len(sim_hist) else None
            if i not in idx_of:
                out[key] = {"in_candidates": False, "in_top10": False, "because_you_played": because,
                            "reasons": [{"group": "similar", "text": f"Similar to {because}" if because else "Explore"}]}
                continue
            j = idx_of[i]
            c = contrib[rows.index(j)][:-1]
            groups, total = {}, 0.0
            for f, v in zip(FEATS, c):
                if f in USER_ONLY or f not in REASON or v <= 0:
                    continue
                total += float(v)
                g, text = REASON[f]
                if g not in groups or v > groups[g][0]:
                    groups[g] = (float(v), text, f)
            reasons = sorted(groups.values(), reverse=True)[:top_reasons]
            feats = {f: float(Xv[j, n]) for n, f in enumerate(FEATS)}
            src = [s for s, ok in (("taste model", feats["in_tt"] > 0), ("trending", feats["in_pop"] > 0)) if ok]
            out[key] = {"in_candidates": True, "in_top10": i in top10, "rank": pos[i], "n_candidates": len(items),
                        "score": round(float(scores[j]), 4), "sources": src, "because_you_played": because,
                        "reasons": [{"group": REASON[f][0], "text": t, "weight": round(v, 3), "feature": f,
                                     "share": round(v / total, 3), "strength": strength(v / total)}
                                    for v, t, f in reasons]}
        return {"source": "personalized", "items": out}
