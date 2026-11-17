"""Streams. start = a play_sessions row + the library upsert in ONE transaction; end = its length.

Idempotent by client_event_id (made by the browser): a retried start or end request changes nothing and
returns the same answer as the first one. `first_time` is read from the data, not remembered: the library
row was created by THIS play exactly when library.added_at = play.started_at, so a retry reports what the
original request reported. started_at is clock_timestamp() (the statement's real time), NOT now(): now() is
the transaction's start, identical for every play inside one transaction, which made a second play look like
the first (found by tests/app_api/test_play_service.py).
"""
import uuid

from sqlalchemy import Text, and_, func, literal, select, true, update
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as DB

from app_api.models import Game, ItemType, LibraryItem, LibrarySource, PlaySession
from app_api.schemas.play import PlayEndIn, PlayOut, PlayStartIn
from app_api.services.errors import Conflict, Invalid, NotFound
from app_api.services.tx import transaction


def _out(db: DB, p: PlaySession) -> PlayOut:
    created_with_play = db.scalar(select(LibraryItem.added_at == p.started_at).where(
        LibraryItem.user_id == p.user_id, LibraryItem.item_id == p.item_id, LibraryItem.source == LibrarySource.play))
    return PlayOut(play_id=p.id, item_id=p.item_id, started_at=p.started_at, ended_at=p.ended_at,
                   seconds_played=p.seconds_played, first_time=bool(created_with_play))


def _own_play(db: DB, user_id: uuid.UUID, client_event_id: uuid.UUID) -> PlaySession:
    p = db.scalar(select(PlaySession).where(PlaySession.client_event_id == client_event_id))
    if p is None:
        raise NotFound("no such play")
    if p.user_id != user_id:
        raise Conflict("client_event_id belongs to another user")
    return p


PLAYABLE = (ItemType.game, ItemType.dlc)


def start_statement(user_id: uuid.UUID, data: PlayStartIn):
    """The whole start as ONE statement (data-modifying CTEs), so it is atomic on its own and costs one round
    trip, even on an autocommit session (Neon: about 263 ms instead of about 1.6 s for 6 round trips):

      p: INSERT play_sessions  SELECT ... FROM games WHERE item is an active game/DLC
                               ON CONFLICT (client_event_id) DO NOTHING          RETURNING id, started_at
      l: INSERT library        SELECT ... FROM p
                               ON CONFLICT (user, item) DO UPDATE play_count + 1 RETURNING added_at, source
      SELECT p.*, first_time = (l.added_at = p.started_at AND l.source = 'play')

    No row back = not playable, unknown, or a retried request (handled by the slow path in start())."""
    src = (select(literal(user_id, UUID(as_uuid=True)), Game.item_id, literal(data.client_event_id, UUID(as_uuid=True)),
                  literal(data.request_id, Text), literal(data.surface, Text), func.clock_timestamp())
           .where(Game.item_id == data.item_id, Game.is_active, Game.item_type.in_(PLAYABLE)))
    p = (pg_insert(PlaySession)
         .from_select(["user_id", "item_id", "client_event_id", "request_id", "surface", "started_at"], src)
         .on_conflict_do_nothing(index_elements=["client_event_id"])
         .returning(PlaySession.id, PlaySession.user_id, PlaySession.item_id, PlaySession.started_at)).cte("p")
    lib = pg_insert(LibraryItem).from_select(
        ["user_id", "item_id", "source", "added_at", "last_played_at", "play_count"],
        select(p.c.user_id, p.c.item_id, literal(LibrarySource.play, LibraryItem.__table__.c.source.type),
               p.c.started_at, p.c.started_at, literal(1)))
    lib_cte = (lib.on_conflict_do_update(index_elements=["user_id", "item_id"],
                                         set_={"last_played_at": lib.excluded.last_played_at,
                                               "play_count": LibraryItem.play_count + 1})
               .returning(LibraryItem.added_at, LibraryItem.source)).cte("l")
    first_time = and_(lib_cte.c.added_at == p.c.started_at, lib_cte.c.source == LibrarySource.play)
    return select(p.c.id, p.c.item_id, p.c.started_at, first_time.label("first_time")).select_from(p.join(lib_cte, true()))


def start(db: DB, user_id: uuid.UUID, data: PlayStartIn) -> PlayOut:
    """Happy path: exactly ONE statement (tests pin this). Safe on get_autocommit_db."""
    with transaction(db):
        row = db.execute(start_statement(user_id, data)).one_or_none()
        if row is not None:
            return PlayOut(play_id=row.id, item_id=row.item_id, started_at=row.started_at, ended_at=None,
                           seconds_played=None, first_time=bool(row.first_time))
        # slow path (rare): a retried request, or an item that cannot be played
        existing = db.scalar(select(PlaySession).where(PlaySession.client_event_id == data.client_event_id))
        if existing is not None:
            return _out(db, _own_play(db, user_id, data.client_event_id))
        if db.get(Game, data.item_id) is None:
            raise NotFound(f"unknown item {data.item_id}")
        raise Invalid(f"{data.item_id} is not a playable game")


def end(db: DB, user_id: uuid.UUID, data: PlayEndIn) -> PlayOut:
    """Close the stream once; its seconds are added to the library. A repeated end changes nothing."""
    with transaction(db):
        play = db.scalar(update(PlaySession)
                         .where(PlaySession.client_event_id == data.client_event_id, PlaySession.user_id == user_id,
                                PlaySession.ended_at.is_(None))
                         .values(ended_at=func.greatest(func.clock_timestamp(), PlaySession.started_at),
                                 seconds_played=data.seconds)
                         .returning(PlaySession))
        if play is None:
            return _out(db, _own_play(db, user_id, data.client_event_id))
        db.execute(update(LibraryItem)
                   .where(and_(LibraryItem.user_id == user_id, LibraryItem.item_id == play.item_id))
                   .values(seconds_played=LibraryItem.seconds_played + data.seconds))
        return _out(db, play)
