import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base
from app_api.models.enums import LibrarySource, pg_enum

if TYPE_CHECKING:
    from app_api.models.game import Game
    from app_api.models.user import User


class LibraryItem(Base):
    """library: current state per (user, game); onboarding picks and plays. The history sent to the engine."""
    __tablename__ = "library"
    __table_args__ = (
        CheckConstraint("play_count >= 0", name="library_play_count_check"),
        CheckConstraint("seconds_played >= 0", name="library_seconds_played_check"),
        # written the way Postgres reports it back, so Alembic's text comparison sees no change
        Index("library_recent", "user_id", text("COALESCE(last_played_at, added_at) DESC")),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    item_id: Mapped[str] = mapped_column(Text, ForeignKey("games.item_id"), primary_key=True)
    source: Mapped[LibrarySource] = mapped_column(pg_enum(LibrarySource, "library_source"))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    last_played_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    play_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    seconds_played: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))

    user: Mapped["User"] = relationship(back_populates="library")
    game: Mapped["Game"] = relationship()
