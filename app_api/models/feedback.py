import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base
from app_api.models.enums import FeedbackKind, pg_enum

if TYPE_CHECKING:
    from app_api.models.game import Game
    from app_api.models.user import User


class ItemFeedback(Base):
    """item_feedback: one current explicit signal per (user, game); serving drops dislike / not_interested."""
    __tablename__ = "item_feedback"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    item_id: Mapped[str] = mapped_column(Text, ForeignKey("games.item_id"), primary_key=True)
    kind: Mapped[FeedbackKind] = mapped_column(pg_enum(FeedbackKind, "feedback_kind"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    user: Mapped["User"] = relationship(back_populates="feedback")
    game: Mapped["Game"] = relationship()
