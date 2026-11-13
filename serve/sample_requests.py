"""
Prints sample user_ids / item_ids from YOUR gold data + ready-to-paste requests for the running API.

    python -m serve.sample_requests                 # default port 8000
    python -m serve.sample_requests --search mario  # items whose title contains "mario"

User types:
  heavy    many games in history         -> personalized
  light    1-2 games in history          -> personalized (thin signal)
  no_hist  only in test window           -> fallback (popular list)
"""
import argparse

import numpy as np
import polars as pl

from serve.config import ServeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--search", default=None, help="title keyword to find item_ids")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    G = ServeConfig.from_env().gold
    hist = pl.concat([pl.read_parquet(G / "train.parquet"), pl.read_parquet(G / "val.parquet")])
    test = pl.read_parquet(G / "test.parquet")
    umap = pl.read_parquet(G / "user_map.parquet")
    items = pl.read_parquet(G / "item_features.parquet").select("iid", "item_id", "title")

    counts = hist.group_by("uid").len()
    take = lambda df: df.sample(min(a.n, df.height), seed=0)
    heavy = counts.sort("len", descending=True).head(a.n)
    light = take(counts.filter(pl.col("len") <= 2))
    no_hist = take(test.select("uid").unique().join(counts, on="uid", how="anti"))
    pick = lambda df: df.join(umap, on="uid")["user_id"].to_list()
    users = {"heavy": pick(heavy), "light": pick(light), "no_hist": pick(no_hist)}

    pop = hist.group_by("iid").len().sort("len", descending=True).head(5).join(items, on="iid").sort("len", descending=True)
    new = items.join(hist.select("iid").unique(), on="iid", how="anti").head(3)
    base = f"http://localhost:{a.port}"

    print("=== USERS ===")
    for kind, ids in users.items():
        for u in ids:
            n = counts.join(umap.filter(pl.col("user_id") == u), on="uid")["len"]
            print(f"  {kind:<8} {u}   (history: {n[0] if len(n) else 0} games)")
    print("\n=== ITEMS (popular in history) ===")
    for r in pop.iter_rows(named=True):
        print(f"  {r['item_id']}   {r['title'][:70]}")
    print("\n=== ITEMS (new: never in history) ===")
    for r in new.iter_rows(named=True):
        print(f"  {r['item_id']}   {r['title'][:70]}")
    if a.search:
        hits = items.filter(pl.col("title").str.to_lowercase().str.contains(a.search.lower(), literal=True)).head(10)
        print(f"\n=== ITEMS matching '{a.search}' ===")
        for r in hits.iter_rows(named=True):
            print(f"  {r['item_id']}   {r['title'][:70]}")

    first = lambda k: users[k][0] if users[k] else "made_up_user_123"
    h, l, z = first("heavy"), first("light"), first("no_hist")
    it = pop["item_id"][0]
    print(f"""
=== TRY IN BROWSER ===
  {base}/docs
  {base}/health
  {base}/recommend/{h}?k=10&debug=true
  {base}/recommend/{l}?k=10&debug=true
  {base}/recommend/{z}?k=10                 (no history -> source: fallback)

=== POWERSHELL: new-user flow (fallback -> event -> personalized) ===
  Invoke-RestMethod {base}/recommend/demo_user_1 | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Method Post -Uri {base}/event -ContentType "application/json" -Body '{{"user_id":"demo_user_1","item_id":"{it}"}}'
  Invoke-RestMethod {base}/recommend/demo_user_1 | ConvertTo-Json -Depth 5
""")


if __name__ == "__main__":
    main()
