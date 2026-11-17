"""The app's copy of the catalog (`games`), loaded from gold by scripts/load_catalog.py on every model release.

load() is ONE transaction (the app never sees half a catalog) and idempotent:
- upsert in batches; a row is only rewritten when a field really changed (ON CONFLICT ... DO UPDATE WHERE the
  values are DISTINCT), so a second identical load reports 0 inserted, 0 updated;
- items missing from the new catalog are deactivated (is_active = false), never deleted: old plays, library
  rows and feedback keep their foreign key; an item that comes back is active again.
Batches of 1,000 rows: about 26 round trips for 25,612 items instead of one per row (Neon: ~265 ms each).
"""
from sqlalchemy import Text, func, literal, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as DB

from app_api.models import Game
from app_api.schemas.catalog import CatalogLoadOut, GameIn
from app_api.services.errors import Invalid
from app_api.services.tx import transaction

FIELDS = ["title", "title_family", "item_type", "platforms", "publisher", "description", "listed_on"]
BATCH = 1000


def _upsert_batch(db: DB, rows: list[GameIn]) -> tuple[int, int]:
    """-> (inserted, updated) for this batch; the rest were unchanged."""
    stmt = pg_insert(Game).values([{**r.model_dump(), "is_active": True} for r in rows])
    ex = stmt.excluded
    changed = or_(*[getattr(Game, f).is_distinct_from(getattr(ex, f)) for f in FIELDS], Game.is_active.is_(False))
    stmt = stmt.on_conflict_do_update(
        index_elements=["item_id"],
        set_={**{f: getattr(ex, f) for f in FIELDS}, "is_active": True, "updated_at": func.now()},
        where=changed,
    ).returning(literal_column("(xmax = 0)").label("inserted"))   # xmax = 0: the row was inserted, not updated
    flags = db.scalars(stmt).all()
    return sum(flags), len(flags) - sum(flags)


def load(db: DB, rows: list[GameIn], batch: int = BATCH) -> CatalogLoadOut:
    ids = [r.item_id for r in rows]
    if len(set(ids)) != len(ids):
        raise Invalid("duplicate item_id in the catalog")
    inserted = updated = 0
    with transaction(db):
        for i in range(0, len(rows), batch):
            a, b = _upsert_batch(db, rows[i: i + batch])
            inserted, updated = inserted + a, updated + b
        deactivated = db.execute(
            update(Game).where(Game.is_active, ~(Game.item_id == func.any(literal(ids, ARRAY(Text)))))
            .values(is_active=False, updated_at=func.now())).rowcount
    return CatalogLoadOut(rows=len(rows), inserted=inserted, updated=updated,
                          unchanged=len(rows) - inserted - updated, deactivated=deactivated)


def counts(db: DB) -> dict[str, int]:
    """Active rows per item_type (for checks and the loader's summary)."""
    return {t.value: n for t, n in db.execute(select(Game.item_type, func.count()).where(Game.is_active)
                                              .group_by(Game.item_type))}
