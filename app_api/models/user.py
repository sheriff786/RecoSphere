import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base
from app_api.models.enums import CIText, UserKind, UserRole, pg_enum

if TYPE_CHECKING:
    from app_api.models.feedback import ItemFeedback
    from app_api.models.library import LibraryItem
    from app_api.models.play import PlaySession
    from app_api.models.session import Session


class User(Base):
    """users: one row per person who uses the app (registered accounts and demo players)."""
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("kind <> 'registered' OR (email IS NOT NULL AND password_hash IS NOT NULL)",
                        name="registered_has_credentials"),
        CheckConstraint("kind <> 'demo' OR dataset_user_id IS NOT NULL", name="demo_has_dataset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    kind: Mapped[UserKind] = mapped_column(pg_enum(UserKind, "user_kind"), server_default=text("'registered'"))
    email: Mapped[str | None] = mapped_column(CIText(), unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text)                 # bcrypt; never the password
    display_name: Mapped[str] = mapped_column(Text)
    role: Mapped[UserRole] = mapped_column(pg_enum(UserRole, "user_role"), server_default=text("'player'"))
    dataset_user_id: Mapped[str | None] = mapped_column(Text, unique=True)  # Amazon reviewer id, demo players only
    onboarded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # NULL = needs onboarding
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # the database deletes these rows (ON DELETE CASCADE); passive_deletes lets it do so
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", passive_deletes=True)
    library: Mapped[list["LibraryItem"]] = relationship(back_populates="user", passive_deletes=True)
    plays: Mapped[list["PlaySession"]] = relationship(back_populates="user", passive_deletes=True)
    feedback: Mapped[list["ItemFeedback"]] = relationship(back_populates="user", passive_deletes=True)

    def __repr__(self) -> str:
        return f"User(id={self.id}, kind={self.kind.value if self.kind else None}, email={self.email!r})"
