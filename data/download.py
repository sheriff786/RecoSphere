"""
RankForge — Phase 0, step 1: Amazon Reviews 2023 download + Parquet conversion.

Usage:
    pip install -U huggingface_hub polars
    python download.py --category Video_Games            # L1
    python download.py --category Movies_and_TV          # L2 (later)

Output (data/raw/<category>/):
    interactions.parquet   user_id, item_id, rating, ts (seconds), verified
    items.parquet          item_id, title, description, features, categories, store, price, ...
    benchmark/             official 5-core timestamp split (sanity-check only)
"""
import argparse
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "McAuley-Lab/Amazon-Reviews-2023"


def find(files, *must_contain):
    """Repo layout hardcode nahi kar rahe — file list mein se match dhoondo."""
    hits = [f for f in files if all(s in f for s in must_contain)]
    if not hits:
        raise FileNotFoundError(f"No file matching {must_contain}")
    return hits


def get(path, out_dir):
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=out_dir)


def main(category: str, root: str = "data/raw"):
    out = Path(root) / category
    out.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(REPO, repo_type="dataset")

    # ---------- 1. Raw reviews -> interactions.parquet ----------
    review_path = [f for f in find(files, "raw/review_categories", category) if f.endswith(".jsonl")][0]
    print("downloading", review_path)
    local = get(review_path, out / "_cache")

    review_schema = {
        "user_id": pl.Utf8,
        "parent_asin": pl.Utf8,
        "rating": pl.Float32,
        "timestamp": pl.Int64,
        "verified_purchase": pl.Boolean,
    }
    (
        pl.scan_ndjson(local, schema=review_schema)          # review text skip -> RAM bachao
        .select(
            pl.col("user_id"),
            pl.col("parent_asin").alias("item_id"),          # variants collapse; meta isi key par hai
            pl.col("rating"),
            (pl.col("timestamp") // 1000).alias("ts"),       # ms -> seconds (IMPORTANT)
            pl.col("verified_purchase").alias("verified"),
        )
        .sink_parquet(out / "interactions.parquet")
    )

    # ---------- 2. Item metadata -> items.parquet ----------
    meta_path = [f for f in find(files, "raw/meta_categories", category) if f.endswith(".jsonl")][0]
    print("downloading", meta_path)
    local_meta = get(meta_path, out / "_cache")
    meta = pl.read_ndjson(local_meta, infer_schema_length=None)
    keep = [c for c in ["parent_asin", "title", "description", "features", "categories",
                        "store", "price", "average_rating", "rating_number", "main_category"]
            if c in meta.columns]
    meta = meta.select(keep).rename({"parent_asin": "item_id"})
    # price kabhi "None"/string hota hai -> float, warna null
    if "price" in meta.columns:
        meta = meta.with_columns(pl.col("price").cast(pl.Utf8).cast(pl.Float64, strict=False))
    meta.write_parquet(out / "items.parquet")

    # ---------- 3. Official 5-core timestamp split (sanity check) ----------
    for f in find(files, "benchmark/5core/timestamp_w_his", category):
        print("downloading", f)
        get(f, out / "benchmark")

    # ---------- Quick stats ----------
    inter = pl.read_parquet(out / "interactions.parquet")
    print(inter.select(
        pl.len().alias("interactions"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("item_id").n_unique().alias("items"),
        pl.from_epoch("ts").min().alias("first"),
        pl.from_epoch("ts").max().alias("last"),
    ))
    per_user = inter.group_by("user_id").len()
    print("users with exactly 1 interaction:",
          f"{(per_user['len'] == 1).mean():.1%}")
    print("items with metadata:",
          f"{inter['item_id'].unique().is_in(meta['item_id'].unique().implode()).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Video_Games")
    args = ap.parse_args()
    main(args.category)
