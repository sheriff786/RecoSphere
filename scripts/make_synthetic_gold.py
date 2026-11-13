"""
Synthetic gold data for CI and smoke tests (Phase 7g): everything the pipeline + API need, in the same
schema notebooks 02/03 write, without Spark, downloads or notebooks. Numbers from this data mean nothing.

    python scripts/make_synthetic_gold.py --out data_ci --users 3000 --items 1200

Writes <out>/gold/Video_Games/{train,val,test,user_map,item_map,item_features}.parquet + item_text_emb_tfidf.npy
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

WORDS = [["quest", "dragon", "sword"], ["race", "car", "speed"], ["soccer", "goal", "league"],
         ["shooter", "war", "squad"], ["puzzle", "block", "brain"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_ci")
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--items", type=int, default=1200)
    ap.add_argument("--clusters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    I, U, C = a.items, a.users, a.clusters
    item_c = rng.integers(0, C, I)
    t0 = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
    t1 = datetime(2023, 9, 1, tzinfo=timezone.utc).timestamp()
    release = rng.uniform(t0 - 3e8, t1, I)
    quality = rng.lognormal(0, 1, I)

    rows = []
    for u in range(U):                                   # users like one cluster + recent, good items
        c = rng.integers(0, C)
        n = 5 + int(rng.zipf(1.7)) % 30
        for t in np.sort(rng.uniform(t0, t1, n)):
            pool = np.flatnonzero((release <= t) & ((item_c == c) | (rng.random(I) < 0.02)))
            if len(pool) == 0:
                continue
            w = quality[pool] * np.exp(-(t - release[pool]) / 86400 / 400) * np.where(item_c[pool] == c, 1.0, 0.2)
            rows.append((u, int(rng.choice(pool, p=w / w.sum())), int(t)))
    df = pl.DataFrame(rows, schema=["uid", "iid", "ts"], orient="row").unique(["uid", "iid"])
    items = np.sort(df["iid"].unique().to_numpy()); users = np.sort(df["uid"].unique().to_numpy())
    imap = {o: n for n, o in enumerate(items)}; umap = {o: n for n, o in enumerate(users)}
    df = df.with_columns(pl.col("iid").replace_strict(imap).cast(pl.Int32), pl.col("uid").replace_strict(umap).cast(pl.Int32),
                         pl.lit(5.0).cast(pl.Float32).alias("rating")).sort("ts")
    tv, tt = (int(df["ts"].quantile(q, interpolation="lower")) for q in (0.8, 0.9))
    g = Path(a.out) / "gold" / "Video_Games"; g.mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("ts") < tv).write_parquet(g / "train.parquet")
    df.filter((pl.col("ts") >= tv) & (pl.col("ts") < tt)).write_parquet(g / "val.parquet")
    df.filter(pl.col("ts") >= tt).write_parquet(g / "test.parquet")
    pl.DataFrame({"uid": np.arange(len(users), dtype=np.int32), "user_id": [f"U{x}" for x in users]}).write_parquet(g / "user_map.parquet")
    pl.DataFrame({"iid": np.arange(len(items), dtype=np.int32), "item_id": [f"P{x}" for x in items]}).write_parquet(g / "item_map.parquet")

    titles = [f"{WORDS[item_c[i] % 5][0].title()} {WORDS[item_c[i] % 5][1].title()} genre{item_c[i]} edition {i}" for i in items]
    texts = [f"{t} | {WORDS[item_c[i] % 5][2]} mode multiplayer" for t, i in zip(titles, items)]
    feat = pl.DataFrame({"iid": np.arange(len(items), dtype=np.int32), "item_id": [f"P{x}" for x in items],
                         "title": titles, "text": texts,
                         "store_id": rng.integers(0, 6, len(items)), "type_id": rng.integers(0, 4, len(items)),
                         "rated_id": rng.integers(0, 4, len(items)), "dfa_ts": release[items].astype(np.int64)})
    feat.write_parquet(g / "item_features.parquet")
    X = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    E = TruncatedSVD(n_components=min(64, X.shape[1] - 1), random_state=0).fit_transform(X)
    np.save(g / "item_text_emb_tfidf.npy", (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)).astype(np.float32))
    print(f"synthetic gold -> {g}: {df.height:,} rows, {len(users):,} users, {len(items):,} items")


if __name__ == "__main__":
    main()
