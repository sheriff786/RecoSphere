"""catalog_service.load: one transaction, idempotent, deactivates instead of deleting."""
import pytest
from sqlalchemy import select

from app_api.models import Game, ItemType
from app_api.schemas.catalog import GameIn
from app_api.services import catalog_service as catalog
from app_api.services.errors import Invalid


def rows(n: int, prefix="CAT-") -> list[GameIn]:
    return [GameIn(item_id=f"{prefix}{i}", title=f"Game {i} - PC", title_family=f"game {i}",
                   item_type=ItemType.game if i % 4 else ItemType.accessory, platforms=["PC"]) for i in range(n)]


def active(db, item_id) -> bool:
    db.expire_all()
    return db.scalar(select(Game.is_active).where(Game.item_id == item_id))


def test_load_is_idempotent_and_counts_changes(db):
    first = catalog.load(db, rows(25), batch=10)                 # 3 batches
    assert (first.inserted, first.updated, first.unchanged) == (25, 0, 0)
    again = catalog.load(db, rows(25), batch=10)
    assert (again.inserted, again.updated, again.unchanged, again.deactivated) == (0, 0, 25, 0)
    changed = rows(25)
    changed[3] = changed[3].model_copy(update={"title": "Game 3 Remastered - PC"})
    third = catalog.load(db, changed, batch=10)
    assert (third.inserted, third.updated, third.unchanged) == (0, 1, 24)


def test_missing_items_are_deactivated_not_deleted_and_come_back(db):
    catalog.load(db, rows(5))
    out = catalog.load(db, rows(3))                              # CAT-3, CAT-4 left the catalog
    assert out.deactivated == 2
    assert active(db, "CAT-4") is False                          # still there: plays keep their foreign key
    back = catalog.load(db, rows(5))
    assert back.updated == 2 and active(db, "CAT-4") is True


def test_seed_games_outside_the_load_are_deactivated_too(db):
    catalog.load(db, rows(2))                                    # the test seed games are not in this catalog
    assert active(db, "G-ZELDA") is False


def test_load_is_one_transaction(db):
    catalog.load(db, rows(4))
    with pytest.raises(Invalid):
        catalog.load(db, rows(4) + rows(1))                      # duplicate id: rejected before any write
    bad = rows(6)
    bad[0] = bad[0].model_copy(update={"title": "Changed in batch 1"})   # batch 1 succeeds ...
    bad[5] = bad[5].model_copy(update={"title": None})                    # ... batch 3 hits NOT NULL
    with pytest.raises(Exception):
        catalog.load(db, bad, batch=2)
    db.expire_all()
    assert db.scalar(select(Game.title).where(Game.item_id == "CAT-0")) == "Game 0 - PC"   # batch 1 rolled back too
    assert db.scalar(select(Game.item_id).where(Game.item_id == "CAT-4")) is None


def test_counts_by_type(db):
    catalog.load(db, rows(8))
    assert catalog.counts(db) == {"game": 6, "accessory": 2}
