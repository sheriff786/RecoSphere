"""Fill the app database's `games` table from the gold catalog (P1 phase C). Run after every new data / model
release; it is idempotent (a second run reports 0 inserted, 0 updated).

    python scripts/load_catalog.py              # writes to DATABASE_URL_DIRECT (app_api/.env or env)   (make load-catalog)
    python scripts/load_catalog.py --dry-run    # build and check the rows, write nothing

Sources (all regenerable, under data/):
  gold/<cat>/item_features.parquet   title, text (category path, description), dfa_ts (date first available)
  gold/<cat>/item_types.parquet      game | accessory | other (data/item_types.py; classified here if missing)
  silver/products/category=<cat>     `store` -> publisher (optional: NULL when the silver layer is absent)
Item types: the classifier has 3, the schema 5: game->game, accessory->accessory, other->other; `console` and
`dlc` stay unused until the classifier separates them.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_api.models import ItemType  # noqa: E402
from app_api.schemas.catalog import DESCRIPTION_MAX, GameIn  # noqa: E402
from data.catalog_fields import description, first_available, platforms  # noqa: E402
from data.item_types import load_item_types  # noqa: E402
from data.title_family import families  # noqa: E402

TYPE_MAP = {"game": ItemType.game, "accessory": ItemType.accessory, "other": ItemType.other}


def build_rows(features: pl.DataFrame, types: list[str], publisher: dict[str, str | None]) -> list[GameIn]:
    """Pure: gold item_features (sorted by iid) + one item_type per row + item_id -> publisher  ->  GameIn rows."""
    ids = features["item_id"].to_list()
    titles = features["title"].fill_null("").to_list()
    texts = features["text"].fill_null("").to_list() if "text" in features.columns else [""] * len(ids)
    dfa = features["dfa_ts"].fill_null(0).to_list() if "dfa_ts" in features.columns else [0] * len(ids)
    listed = [first_available(int(d)) for d in dfa]
    fam = families(titles, [d.year if d else None for d in listed])
    return [GameIn(item_id=i, title=t, title_family=f, item_type=TYPE_MAP[ty], platforms=platforms(t, x),
                   publisher=(publisher.get(i) or None), description=description(x, DESCRIPTION_MAX) or None,
                   listed_on=d)
            for i, t, x, f, ty, d in zip(ids, titles, texts, fam, types, listed)]


def publishers(data_root: str, category: str) -> dict[str, str | None]:
    files = glob.glob(f"{data_root}/silver/products/category={category}/*.parquet")
    if not files:
        return {}
    s = pl.concat([pl.read_parquet(f, columns=["item_id", "store"]) for f in files]).unique("item_id")
    return dict(zip(s["item_id"].to_list(), s["store"].to_list()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--data", default="data")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    t0 = time.perf_counter()
    gold = Path(a.data) / "gold" / a.category
    feats = pl.read_parquet(gold / "item_features.parquet").sort("iid")
    types, source = load_item_types(gold, feats["item_id"].to_list(), feats["title"].fill_null("").to_list(),
                                    feats["text"].fill_null("").to_list())
    rows = build_rows(feats, types, publishers(a.data, a.category))
    by_type = pl.Series([r.item_type.value for r in rows]).value_counts(sort=True)
    print(f"built {len(rows):,} rows in {time.perf_counter() - t0:.1f}s | types from {source}")
    print("  " + ", ".join(f"{t}: {n:,}" for t, n in by_type.iter_rows()))
    print(f"  title families: {len({r.title_family for r in rows}):,} | with publisher: "
          f"{sum(r.publisher is not None for r in rows):,} | with listed_on: {sum(r.listed_on is not None for r in rows):,}")
    if a.dry_run:
        return 0

    from sqlalchemy.orm import Session

    from app_api.config import get_settings
    from app_api.db.session import make_engine
    from app_api.services import catalog_service

    engine = make_engine(get_settings().migrations_url, pool_size=1)
    t1 = time.perf_counter()
    with Session(engine) as db:
        out = catalog_service.load(db, rows)
        active = catalog_service.counts(db)
    engine.dispose()
    print(f"loaded in {time.perf_counter() - t1:.1f}s: {out.model_dump()}")
    print(f"active games by type in the database: {active}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
