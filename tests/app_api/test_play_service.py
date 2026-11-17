"""play_service: one transaction per start, idempotent by client_event_id, first_time from the data."""
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from app_api.models import LibraryItem, LibrarySource, PlaySession, User, UserKind
from app_api.schemas.library import PicksIn
from app_api.schemas.play import PlayEndIn, PlayStartIn
from app_api.services import library_service as lib
from app_api.services import play_service as plays
from app_api.services.errors import Conflict, Invalid, NotFound


def new_user(db) -> uuid.UUID:
    u = User(kind=UserKind.registered, email=f"{uuid.uuid4().hex[:8]}@example.test", password_hash="x",
             display_name="T")
    db.add(u)
    db.commit()        # like a real app: the user exists from an earlier request
    return u.id


def lib_row(db, uid, item) -> LibraryItem:
    db.expire_all()
    return db.scalar(select(LibraryItem).where(LibraryItem.user_id == uid, LibraryItem.item_id == item))


def play_count(db, cid) -> int:
    return db.scalar(select(func.count()).select_from(PlaySession).where(PlaySession.client_event_id == cid))


def test_first_play_retry_and_second_play(db, statements):
    uid, cid = new_user(db), uuid.uuid4()
    statements.clear()
    first = plays.start(db, uid, PlayStartIn(item_id="G-ZELDA", client_event_id=cid, request_id="r1",
                                             surface="home:top_picks"))
    assert len(statements) == 1, statements                 # ONE statement: atomic alone, safe on autocommit
    assert first.first_time and first.ended_at is None
    retry = plays.start(db, uid, PlayStartIn(item_id="G-ZELDA", client_event_id=cid))
    assert retry.play_id == first.play_id and retry.first_time   # same answer as the original request
    assert play_count(db, cid) == 1 and lib_row(db, uid, "G-ZELDA").play_count == 1
    second = plays.start(db, uid, PlayStartIn(item_id="G-ZELDA", client_event_id=uuid.uuid4()))
    assert not second.first_time and lib_row(db, uid, "G-ZELDA").play_count == 2
    p = db.get(PlaySession, first.play_id)
    assert (p.request_id, p.surface) == ("r1", "home:top_picks")


def test_playing_an_onboarding_pick_is_not_first_time(db):
    uid = new_user(db)
    lib.add_picks(db, uid, PicksIn(item_ids=["G-ZELDA", "G-MARIO", "G-POKEMON"]))
    out = plays.start(db, uid, PlayStartIn(item_id="G-MARIO", client_event_id=uuid.uuid4()))
    row = lib_row(db, uid, "G-MARIO")
    assert not out.first_time and row.source == LibrarySource.onboarding and row.play_count == 1


@pytest.mark.parametrize("item,error", [("A-PAD", Invalid), ("G-OLD", Invalid), ("NO-SUCH", NotFound)])
def test_only_active_games_can_be_played(db, item, error):
    uid, cid = new_user(db), uuid.uuid4()
    with pytest.raises(error):
        plays.start(db, uid, PlayStartIn(item_id=item, client_event_id=cid))
    assert play_count(db, cid) == 0


def test_dlc_is_playable(db):
    assert plays.start(db, new_user(db), PlayStartIn(item_id="G-DLC", client_event_id=uuid.uuid4())).first_time


def test_someone_elses_event_id_is_a_conflict(db):
    a, b, cid = new_user(db), new_user(db), uuid.uuid4()
    plays.start(db, a, PlayStartIn(item_id="G-ZELDA", client_event_id=cid))
    with pytest.raises(Conflict):
        plays.start(db, b, PlayStartIn(item_id="G-ZELDA", client_event_id=cid))
    with pytest.raises(Conflict):
        plays.end(db, b, PlayEndIn(client_event_id=cid, seconds=5))


def test_start_is_atomic_inside_the_database(db):
    """If the library write fails IN POSTGRES, the play row must not stay behind. A trigger (created in the
    test's schema and rolled back with it) makes every library insert fail."""
    uid, cid = new_user(db), uuid.uuid4()
    db.execute(text("""CREATE FUNCTION fail_library() RETURNS trigger LANGUAGE plpgsql AS
                       $$ BEGIN RAISE EXCEPTION 'library write failed'; END $$"""))
    db.execute(text("CREATE TRIGGER fail_library BEFORE INSERT ON library FOR EACH ROW EXECUTE FUNCTION fail_library()"))
    db.commit()
    with pytest.raises(DBAPIError, match="library write failed"):
        plays.start(db, uid, PlayStartIn(item_id="G-ZELDA", client_event_id=cid))
    assert play_count(db, cid) == 0 and lib_row(db, uid, "G-ZELDA") is None


def test_end_adds_seconds_once(db):
    uid, cid = new_user(db), uuid.uuid4()
    plays.start(db, uid, PlayStartIn(item_id="G-POKEMON", client_event_id=cid))
    out = plays.end(db, uid, PlayEndIn(client_event_id=cid, seconds=95))
    assert out.seconds_played == 95 and out.ended_at is not None and out.ended_at >= out.started_at
    again = plays.end(db, uid, PlayEndIn(client_event_id=cid, seconds=500))    # retried / duplicate end
    assert again.seconds_played == 95 and lib_row(db, uid, "G-POKEMON").seconds_played == 95
    with pytest.raises(NotFound):
        plays.end(db, uid, PlayEndIn(client_event_id=uuid.uuid4(), seconds=1))
