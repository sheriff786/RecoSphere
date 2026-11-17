"""library_service: onboarding picks, recent history, stats."""
import uuid

import pytest
from sqlalchemy import select

from app_api.models import LibraryItem, LibrarySource, User, UserKind
from app_api.schemas.library import PicksIn
from app_api.schemas.play import PlayStartIn
from app_api.services import library_service as lib
from app_api.services import play_service as plays
from app_api.services.errors import Invalid


def new_user(db) -> uuid.UUID:
    u = User(kind=UserKind.registered, email=f"{uuid.uuid4().hex[:8]}@example.test", password_hash="x",
             display_name="T")
    db.add(u)
    db.commit()        # like a real app: the user exists from an earlier request
    return u.id


def test_picks_onboard_the_user_in_one_transaction(db):
    uid = new_user(db)
    added = lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", "G-POKEMON"]))
    assert sorted(added) == ["G-MARIO", "G-POKEMON", "G-ZELDA"]
    assert db.get(User, uid).onboarded_at is not None
    again = lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", "G-DLC"]))
    assert again == ["G-DLC"]                                   # only what was new


@pytest.mark.parametrize("bad", ["A-PAD", "G-OLD", "NO-SUCH-ID"])   # accessory, delisted, unknown
def test_a_bad_pick_writes_nothing(db, bad):
    uid = new_user(db)
    with pytest.raises(Invalid, match=bad):
        lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", bad]))
    assert db.scalars(select(LibraryItem).where(LibraryItem.user_id == uid)).all() == []
    assert db.get(User, uid).onboarded_at is None


def test_recent_is_most_recent_first_and_games_only(db, statements):
    uid = new_user(db)
    lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", "G-POKEMON"]))
    plays.start(db, uid, PlayStartIn(item_id="G-MARIO", client_event_id=uuid.uuid4()))
    db.add(LibraryItem(user_id=uid, item_id="A-PAD", source=LibrarySource.play))   # e.g. an old Amazon purchase
    db.flush()
    statements.clear()
    rows = lib.recent(db, uid)
    assert len(statements) == 1, statements
    assert rows[0].item_id == "G-MARIO" and rows[0].play_count == 1
    assert "A-PAD" not in [r.item_id for r in rows]
    assert "A-PAD" in [r.item_id for r in lib.recent(db, uid, games_only=False)]


def test_stats(db):
    uid = new_user(db)
    assert lib.stats(db, uid).model_dump() == {"games": 0, "plays": 0, "seconds_played": 0}
    lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", "G-POKEMON"]))
    assert lib.stats(db, uid).games == 3
