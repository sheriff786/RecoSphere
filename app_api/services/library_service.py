"""The user's library: onboarding picks, recent history (what the engine gets as history), stats."""
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as DB

from app_api.models import Game, ItemType, LibraryItem, LibrarySource, User
from app_api.schemas.library import LibraryItemOut, LibraryStatsOut, PicksIn
from app_api.services.errors import Invalid
from app_api.services.tx import transaction

PLAYABLE = (ItemType.game, ItemType.dlc)
RECENT = func.coalesce(LibraryItem.last_played_at, LibraryItem.added_at)   # = the library_recent index


def add_picks(db: DB, user_id: uuid.UUID, data: PicksIn) -> list[str]:
    """Onboarding: the picks become library rows (source 'onboarding') and the user is onboarded, in one
    transaction. Every pick must be an active game/DLC. Returns the ids that were new to the library."""
    with transaction(db):
        ok = set(db.scalars(select(Game.item_id).where(Game.item_id.in_(data.item_ids), Game.is_active,
                                                       Game.item_type.in_(PLAYABLE))))
        bad = [i for i in data.item_ids if i not in ok]
        if bad:
            raise Invalid(f"not playable games: {bad}")
        added = db.scalars(pg_insert(LibraryItem)
                           .values([{"user_id": user_id, "item_id": i, "source": LibrarySource.onboarding}
                                    for i in data.item_ids])
                           .on_conflict_do_nothing(index_elements=["user_id", "item_id"])
                           .returning(LibraryItem.item_id)).all()
        db.execute(update(User).where(User.id == user_id, User.onboarded_at.is_(None)).values(onboarded_at=func.now()))
    return list(added)


def recent(db: DB, user_id: uuid.UUID, k: int = 20, games_only: bool = True) -> list[LibraryItemOut]:
    """Most recent first (last play, else when it was added): one indexed read joined to games."""
    q = (select(LibraryItem, Game.title).join(Game, Game.item_id == LibraryItem.item_id)
         .where(LibraryItem.user_id == user_id).order_by(RECENT.desc(), LibraryItem.item_id).limit(k))
    if games_only:
        q = q.where(Game.item_type.in_(PLAYABLE))
    return [LibraryItemOut(item_id=li.item_id, title=title, source=li.source, added_at=li.added_at,
                           last_played_at=li.last_played_at, play_count=li.play_count,
                           seconds_played=li.seconds_played) for li, title in db.execute(q)]


def stats(db: DB, user_id: uuid.UUID) -> LibraryStatsOut:
    games, plays, secs = db.execute(
        select(func.count(), func.coalesce(func.sum(LibraryItem.play_count), 0),
               func.coalesce(func.sum(LibraryItem.seconds_played), 0)).where(LibraryItem.user_id == user_id)).one()
    return LibraryStatsOut(games=games, plays=plays, seconds_played=secs)
