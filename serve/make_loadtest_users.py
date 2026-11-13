"""
Writes loadtest/users.txt and loadtest/items.txt for Locust.
Mix: users with history (personalized path) + ~10% made-up ids (fallback path), like real traffic.

    python -m serve.make_loadtest_users --n 5000
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from serve.config import ServeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--unknown-frac", type=float, default=0.1)
    a = ap.parse_args()
    cfg = ServeConfig.from_env()
    hist = pl.concat([pl.read_parquet(cfg.gold / f"{s}.parquet") for s in ("train", "val")])
    umap = pl.read_parquet(cfg.gold / "user_map.parquet").filter(pl.col("uid").is_in(hist["uid"].unique().implode()))
    rng = np.random.default_rng(0)
    known = rng.choice(umap["user_id"].to_numpy(), size=min(a.n, umap.height), replace=False).tolist()
    unknown = [f"new_user_{i}" for i in range(int(a.unknown_frac * len(known)))]
    items = pl.read_parquet(cfg.gold / "item_map.parquet")["item_id"].to_list()
    out = cfg.repo_root / "loadtest"; out.mkdir(exist_ok=True)
    (out / "users.txt").write_text("\n".join(known + unknown) + "\n")
    (out / "items.txt").write_text("\n".join(rng.choice(items, size=min(2000, len(items)), replace=False).tolist()) + "\n")
    print(f"wrote {len(known)} known + {len(unknown)} unknown users, items -> {out}")


if __name__ == "__main__":
    main()
