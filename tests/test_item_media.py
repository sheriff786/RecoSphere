"""data/item_media.py: pictures for the app from the raw Amazon metadata (presentation only, not a model feature)."""
import json
from pathlib import Path

import polars as pl

from data.item_media import build, load_item_media, media_of

A = "https://m.media-amazon.com/images/I/"


def test_main_image_is_the_cover_and_the_rest_is_the_gallery():
    m = media_of([
        {"variant": "PT01", "large": A + "b.jpg", "hi_res": A + "B._SL1500_.jpg"},
        {"variant": "MAIN", "large": A + "a.jpg", "hi_res": A + "A._SL1500_.jpg"},
        {"variant": "PT02", "large": A + "c.jpg", "hi_res": None},
    ])
    assert m == {"cover": A + "a.jpg", "cover_hires": A + "A._SL1500_.jpg",
                 "gallery": [A + "B._SL1500_.jpg", A + "c.jpg"]}       # no hi_res -> large


def test_only_amazon_image_urls_reach_the_browser():
    bad = ["http://m.media-amazon.com/images/I/x.jpg",                   # not https
           "https://evil.example/images/I/x.jpg",                        # other host
           "https://m.media-amazon.com/images/I/x.jpg\" onerror=\"alert(1)",   # attribute injection
           "javascript:alert(1)", None, 42]
    assert media_of([{"variant": "MAIN", "large": u, "hi_res": u} for u in bad]) is None
    m = media_of([{"variant": "MAIN", "large": "javascript:alert(1)", "hi_res": A + "ok.jpg"}])
    assert m == {"cover": A + "ok.jpg", "cover_hires": A + "ok.jpg", "gallery": []}
    assert media_of(None) is None and media_of([]) is None


def test_build_keeps_only_catalog_items(tmp_path):
    meta = tmp_path / "meta.jsonl"
    meta.write_text("\n".join(json.dumps(d) for d in [
        {"parent_asin": "G1", "images": [{"variant": "MAIN", "large": A + "g1.jpg"}]},
        {"parent_asin": "NOT_IN_CATALOG", "images": [{"variant": "MAIN", "large": A + "x.jpg"}]},
        {"parent_asin": "G2", "images": []},
    ]))
    df = build(meta, {"G1", "G2"})
    assert df["item_id"].to_list() == ["G1"] and df["cover"].to_list() == [A + "g1.jpg"]


def test_engine_loader_prefers_the_snapshot_and_falls_back_to_local_gold(tmp_path):
    snap, local = tmp_path / "snapshot", tmp_path / "local"
    snap.mkdir(); local.mkdir()
    pl.DataFrame({"item_id": ["G1"], "cover": [A + "local.jpg"], "cover_hires": [A + "local.jpg"], "gallery": [[]]},
                 schema={"item_id": pl.Utf8, "cover": pl.Utf8, "cover_hires": pl.Utf8,
                         "gallery": pl.List(pl.Utf8)}).write_parquet(local / "item_media.parquet")
    media, src = load_item_media([snap, local], ["G1", "G2"])          # snapshot predates the file
    assert media[0]["cover"] == A + "local.jpg" and media[1] is None
    assert Path(src).parent.name == "local"
    assert load_item_media([snap], ["G1"]) == ([None], "none")          # no file anywhere: generated covers
