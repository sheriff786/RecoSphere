import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base

if TYPE_CHECKING:
    from app_api.models.user import User


class Session(Base):
    """sessions: the server-side record behind the JWT cookie (the JWT carries sid = sessions.id).
    Not to be confused with a SQLAlchemy Session: this is a LOGIN session."""
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="sessions_check"),
        Index("sessions_active_by_user", "user_id", postgresql_where=text("revoked_at IS NULL")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="sessions")
