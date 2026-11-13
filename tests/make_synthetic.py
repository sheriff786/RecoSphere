"""
Fake raw data (download.py jaisa schema) — pipeline ko end-to-end smoke test karne ke liye,
22 GB download kiye bina. Latent clusters + time drift hai, taaki baselines ke beech
farak dikhe. Ye numbers kahin report mat karna — sirf "pipeline chalti hai" ka proof hai.

    python tests/make_synthetic.py && python -m data.prepare --category Synthetic ...
"""
from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(42)
U, I, C = 6000, 2000, 20
user_c = rng.integers(0, C, U)
item_c = rng.integers(0, C, I)
item_pop = rng.zipf(1.6, I).clip(1, 500).astype(float)

rows = []
t0, span = 1_500_000_000, 3 * 365 * 86400
for u in range(U):
    n = int(rng.zipf(1.8)) + 1
    n = min(n, 60)
    ts = np.sort(rng.integers(t0, t0 + span, n))
    for t in ts:
        drift = (t - t0) / span  # time ke saath naye items popular hote hain
        same = rng.random() < 0.7
        pool = np.flatnonzero(item_c == user_c[u]) if same else np.arange(I)
        w = item_pop[pool] * np.exp(3 * drift * (pool / I))
        it = rng.choice(pool, p=w / w.sum())
        rows.append((f"U{u}", f"I{it}", float(rng.integers(1, 6)), int(t), True))

df = pl.DataFrame(rows, schema=["user_id", "item_id", "rating", "ts", "verified"], orient="row")
out = Path("data/raw/Synthetic"); out.mkdir(parents=True, exist_ok=True)
df.write_parquet(out / "interactions.parquet")
print(f"wrote {df.height:,} synthetic interactions -> {out}")
