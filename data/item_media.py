"""
Images per item, for the app only (never a model feature): the gold catalog -> data/gold/<cat>/item_media.parquet.

    python -m data.item_media --write          # re-run after new data (like `python -m data.item_types --write`)

Source: the raw Amazon metadata (bronze meta_<cat>.jsonl, field `images`), matched by item_id (parent_asin), so a
picture is always THIS product's, never a guess from a title search. Per item:
  cover        the MAIN image, `large` (~500 px): cards and rows
  cover_hires  the MAIN image, `hi_res` (1500 px) when Amazon has it, else `large`: the game page
  gallery      up to 8 more hi-res images (other variants: back cover, screenshots ...), in Amazon's order
Only https://m.media-amazon.com/images/ URLs are kept: the engine hands these to browsers, so nothing else gets in.
The images are Amazon's / the publishers' and are linked, not copied; the app falls back to its generated covers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_PREFIX = "https://m.media-amazon.com/images/"
GALLERY_MAX = 8


def _ok(url) -> bool:
    return isinstance(url, str) and url.startswith(ALLOWED_PREFIX) and len(url) < 300 and not any(
        c in url for c in "\"'<> \\")


def media_of(images: list | None) -> dict | None:
    """One item's `images` list (Amazon metadata) -> {cover, cover_hires, gallery} or None when nothing usable."""
    imgs = [im for im in (images or []) if isinstance(im, dict) and (_ok(im.get("large")) or _ok(im.get("hi_res")))]
    if not imgs:
        return None
    main = next((im for im in imgs if im.get("variant") == "MAIN"), imgs[0])
    big = lambda im: im.get("hi_res") if _ok(im.get("hi_res")) else im.get("large")
    small = lambda im: im.get("large") if _ok(im.get("large")) else im.get("hi_res")
    gallery = [big(im) for im in imgs if im is not main][:GALLERY_MAX]
    return {"cover": small(main), "cover_hires": big(main), "gallery": gallery}


def build(meta_jsonl: Path, item_ids: set[str]):
    """Stream the metadata file (hundreds of MB) and keep only the catalog's items."""
    import polars as pl
    rows = []
    with open(meta_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            a = d.get("parent_asin")
            if a in item_ids:
                m = media_of(d.get("images"))
                if m:
                    rows.append({"item_id": a, **m})
    schema = {"item_id": pl.Utf8, "cover": pl.Utf8, "cover_hires": pl.Utf8, "gallery": pl.List(pl.Utf8)}
    return pl.DataFrame(rows, schema=schema).unique("item_id", keep="first").sort("item_id")


def load_item_media(gold_dirs, item_ids: list[str]) -> tuple[list[dict | None], str]:
    """For the engine: media aligned with item_ids (None = no picture) from the first gold dir that has the file.
    It is presentation, not model data, so a run's gold snapshot may predate it: the caller passes the snapshot
    first and the local gold folder second. -> (media per item, source)."""
    import polars as pl
    for d in gold_dirs:
        p = Path(d) / "item_media.parquet" if d else None
        if p is not None and p.exists():
            df = pl.read_parquet(p)
            by_id = {r["item_id"]: r for r in df.iter_rows(named=True) if _ok(r["cover"])}
            out = [None] * len(item_ids)
            for i, x in enumerate(item_ids):
                r = by_id.get(x)
                if r:
                    out[i] = {"cover": r["cover"], "cover_hires": r["cover_hires"] if _ok(r["cover_hires"]) else r["cover"],
                              "gallery": [u for u in (r["gallery"] or []) if _ok(u)]}
            return out, str(p)
    return [None] * len(item_ids), "none"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--data", default="data")
    ap.add_argument("--write", action="store_true", help="write gold/<cat>/item_media.parquet (else: report only)")
    a = ap.parse_args(argv)
    import polars as pl
    gold = Path(a.data) / "gold" / a.category
    meta = Path(a.data) / "bronze" / "metadata" / f"category={a.category}" / f"meta_{a.category}.jsonl"
    feat = pl.read_parquet(gold / "item_features.parquet").select("item_id")
    ids = set(feat["item_id"].to_list())
    df = build(meta, ids)
    types = gold / "item_types.parquet"
    if types.exists():
        t = pl.read_parquet(types).select("item_id", "item_type")
        rep = (t.join(df.select("item_id", pl.col("gallery").list.len().alias("n")), on="item_id", how="left")
               .group_by("item_type").agg(pl.len().alias("items"), pl.col("n").is_not_null().sum().alias("with_cover"),
                                          pl.col("n").mean().round(1).alias("gallery_mean")).sort("items", descending=True))
        print(rep)
    print(f"{df.height:,} of {len(ids):,} catalog items have a cover")
    if a.write:
        df.write_parquet(gold / "item_media.parquet")
        print(f"wrote {gold / 'item_media.parquet'}")


if __name__ == "__main__":
    main()
