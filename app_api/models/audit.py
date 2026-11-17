import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app_api.db.base import Base

if TYPE_CHECKING:
    from app_api.models.user import User


class AdminAuditLog(Base):
    """admin_audit_log: who did what from /admin. Append-only; actor_id becomes NULL if the admin is deleted."""
    __tablename__ = "admin_audit_log"
    __table_args__ = (Index("admin_audit_recent", text("created_at DESC")),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(Text)                                # 'model.rollback', 'experiment.stop'
    target: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    actor: Mapped["User | None"] = relationship()
