"""
Online feature store: per-user state needed at request time.

Per user:
  hist        last `hist_len` item ids, oldest -> newest      (user tower input + text profile)
  seen        every item the user ever interacted with         (masking: never recommend again)
  n_hist, first, last, sum_rating                               (user features)
  stores      publisher -> count                                (ui_store_share feature)

Two backends with the SAME interface:
  InMemoryStore — dev / tests / single process
  RedisStore    — production: survives API restarts, shared by many API replicas

Item-level state (popularity windows, item age) is NOT here: it is rebuilt by the batch job
(Phase 7) and loaded into memory at startup. Events update only user state.
"""
import json

import numpy as np
import polars as pl


def bootstrap_records(hist_df: pl.DataFrame, user_map: pl.DataFrame, store_of: np.ndarray, hist_len: int):
    """Yield (user_id, uid, record) for every user with history — same ordering rule as
    retrieval.two_tower.history_at_cutoff (sort by ts, then iid)."""
    h = hist_df.sort(["uid", "ts", "iid"])
    h = h.with_columns(pl.Series("store", store_of[h["iid"].to_numpy()]))
    g = h.group_by("uid", maintain_order=True).agg(
        pl.col("iid").alias("seen"), pl.col("iid").tail(hist_len).alias("hist"),
        pl.len().alias("n_hist"), pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last"),
        pl.col("rating").cast(pl.Float64).sum().alias("sum_rating"), pl.col("store").alias("stores"))
    names = dict(zip(user_map["uid"].to_list(), user_map["user_id"].to_list()))
    for r in g.iter_rows(named=True):
        st, cnt = np.unique(np.asarray(r["stores"]), return_counts=True)
        yield names[r["uid"]], r["uid"], {
            "hist": list(map(int, r["hist"])), "seen": list(map(int, r["seen"])), "n_hist": int(r["n_hist"]),
            "first": int(r["first"]), "last": int(r["last"]), "sum_rating": float(r["sum_rating"]),
            "stores": {int(s): int(c) for s, c in zip(st, cnt)}}


class InMemoryStore:
    backend = "memory"

    def __init__(self):
        self.umap, self.users, self.next_uid, self.version = {}, {}, 0, None

    def is_loaded(self, version: str) -> bool:
        return self.version == version

    # single process -> no real locking needed
    def acquire_load_lock(self) -> bool:
        return True

    def release_load_lock(self):
        pass

    def wait_until_loaded(self, version: str, timeout: int = 0):
        pass

    def mark_loaded(self, version: str):
        self.version = version

    def load(self, records, n_users: int, version: str):
        for user_id, uid, rec in records:
            self.umap[user_id] = uid
            self.users[uid] = rec
        self.next_uid = n_users

    def register_users(self, user_map: pl.DataFrame):
        # users with NO history (e.g. only in test) still get a stable uid
        for user_id, uid in zip(user_map["user_id"].to_list(), user_map["uid"].to_list()):
            self.umap.setdefault(user_id, uid)

    def get(self, user_id: str):
        uid = self.umap.get(user_id)
        if uid is None or uid not in self.users:
            return uid, None
        r = self.users[uid]
        return uid, {**r, "seen": np.asarray(r["seen"], dtype=np.int64)}

    def add_event(self, user_id: str, iid: int, ts: int, rating: float, store: int, hist_len: int) -> tuple[int, bool]:
        uid = self.umap.get(user_id)
        if uid is None:
            uid = self.next_uid; self.next_uid += 1; self.umap[user_id] = uid
        r = self.users.setdefault(uid, {"hist": [], "seen": [], "n_hist": 0, "first": ts, "last": ts,
                                        "sum_rating": 0.0, "stores": {}})
        if iid in set(r["seen"]):
            return uid, False                                   # dedup rule from notebook 02: first interaction only
        r["hist"] = (r["hist"] + [iid])[-hist_len:]
        r["seen"].append(iid)
        r["n_hist"] += 1; r["first"] = min(r["first"], ts); r["last"] = max(r["last"], ts)
        r["sum_rating"] += rating; r["stores"][store] = r["stores"].get(store, 0) + 1
        return uid, True

    def n_users(self) -> int:
        return len(self.umap)


class RedisStore:
    backend = "redis"

    def __init__(self, url: str, prefix: str):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.p = prefix

    def _k(self, *parts):
        return ":".join([self.p, *map(str, parts)])

    def is_loaded(self, version: str) -> bool:
        return self.r.hget(self._k("meta"), "version") == version

    def acquire_load_lock(self, ttl: int = 1800) -> bool:
        # SET NX = only the first worker gets it; TTL so a crashed loader does not block forever
        return bool(self.r.set(self.p + ":__loadlock", "1", nx=True, ex=ttl))

    def release_load_lock(self):
        self.r.delete(self.p + ":__loadlock")

    def wait_until_loaded(self, version: str, timeout: int = 1800):
        import time
        t0 = time.time()
        while not self.is_loaded(version):
            if time.time() - t0 > timeout:
                raise TimeoutError("feature store was not loaded by the other worker")
            time.sleep(1)

    def mark_loaded(self, version: str):
        self.r.hset(self._k("meta"), "version", version)

    def _clear(self):
        for key in self.r.scan_iter(f"{self.p}:*", count=5000):
            if not key.endswith(":__loadlock"):
                self.r.unlink(key)

    def load(self, records, n_users: int, version: str, batch: int = 2000):
        self._clear()
        pipe, n = self.r.pipeline(transaction=False), 0
        for user_id, uid, rec in records:
            pipe.hset(self._k("umap"), user_id, uid)
            pipe.hset(self._k("u", uid), mapping={"n_hist": rec["n_hist"], "first": rec["first"],
                                                  "last": rec["last"], "sum_rating": rec["sum_rating"]})
            pipe.rpush(self._k("u", uid, "hist"), *rec["hist"])
            pipe.sadd(self._k("u", uid, "seen"), *rec["seen"])
            pipe.hset(self._k("u", uid, "stores"), mapping={str(s): c for s, c in rec["stores"].items()})
            n += 1
            if n % batch == 0:
                pipe.execute()
        pipe.set(self._k("uid_next"), n_users)
        pipe.hset(self._k("meta"), mapping={"users_loaded": n})         # version is set by mark_loaded()
        pipe.execute()

    def register_users(self, user_map: pl.DataFrame, batch: int = 20000):
        ids, uids = user_map["user_id"].to_list(), user_map["uid"].to_list()
        for s in range(0, len(ids), batch):
            pipe = self.r.pipeline(transaction=False)
            for a, b in zip(ids[s:s + batch], uids[s:s + batch]):
                pipe.hsetnx(self._k("umap"), a, b)
            pipe.execute()

    def get(self, user_id: str):
        uid = self.r.hget(self._k("umap"), user_id)
        if uid is None:
            return None, None
        uid = int(uid)
        pipe = self.r.pipeline(transaction=False)
        pipe.hgetall(self._k("u", uid)); pipe.lrange(self._k("u", uid, "hist"), 0, -1)
        pipe.smembers(self._k("u", uid, "seen")); pipe.hgetall(self._k("u", uid, "stores"))
        h, hist, seen, stores = pipe.execute()
        if not h:
            return uid, None
        return uid, {"hist": [int(x) for x in hist], "seen": np.fromiter((int(x) for x in seen), dtype=np.int64),
                     "n_hist": int(h["n_hist"]), "first": int(h["first"]), "last": int(h["last"]),
                     "sum_rating": float(h["sum_rating"]), "stores": {int(k): int(v) for k, v in stores.items()}}

    def add_event(self, user_id: str, iid: int, ts: int, rating: float, store: int, hist_len: int) -> tuple[int, bool]:
        uid = self.r.hget(self._k("umap"), user_id)
        if uid is None:
            uid = int(self.r.incr(self._k("uid_next"))) - 1
            self.r.hset(self._k("umap"), user_id, uid)
        uid = int(uid)
        if not self.r.sadd(self._k("u", uid, "seen"), iid):
            return uid, False                                   # already seen -> dedup
        pipe = self.r.pipeline(transaction=True)
        pipe.rpush(self._k("u", uid, "hist"), iid); pipe.ltrim(self._k("u", uid, "hist"), -hist_len, -1)
        pipe.hincrby(self._k("u", uid), "n_hist", 1)
        pipe.hincrbyfloat(self._k("u", uid), "sum_rating", rating)
        pipe.hsetnx(self._k("u", uid), "first", ts)
        pipe.hincrby(self._k("u", uid, "stores"), str(store), 1)
        pipe.execute()
        last = self.r.hget(self._k("u", uid), "last")
        if last is None or int(last) < ts:
            self.r.hset(self._k("u", uid), "last", ts)
        return uid, True

    def n_users(self) -> int:
        return int(self.r.hlen(self._k("umap")))
