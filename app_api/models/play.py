import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base

if TYPE_CHECKING:
    from app_api.models.game import Game
    from app_api.models.user import User


class PlaySession(Base):
    """play_sessions: append-only history of every stream, with attribution (request_id, surface)."""
    __tablename__ = "play_sessions"
    __table_args__ = (
        CheckConstraint("seconds_played >= 0", name="play_sessions_seconds_played_check"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="play_sessions_check"),
        Index("play_sessions_by_user", "user_id", text("started_at DESC")),
        Index("play_sessions_by_item", "item_id", text("started_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    item_id: Mapped[str] = mapped_column(Text, ForeignKey("games.item_id"))
    client_event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)   # idempotency key
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    seconds_played: Mapped[int | None] = mapped_column(Integer)              # NULL until the session ends
    request_id: Mapped[str | None] = mapped_column(Text)                     # engine request that showed the card
    surface: Mapped[str | None] = mapped_column(Text)                        # e.g. 'home:top_picks', 'search'

    user: Mapped["User"] = relationship(back_populates="plays")
    game: Mapped["Game"] = relationship()
