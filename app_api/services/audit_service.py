"""Admin audit log: an append-only record of what admins did (rollback, experiment stop, role change)."""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session as DB

from app_api.models import AdminAuditLog
from app_api.schemas.audit import AuditIn, AuditOut
from app_api.services.tx import transaction


def record(db: DB, actor_id: uuid.UUID | None, data: AuditIn) -> AuditOut:
    with transaction(db):
        row = AdminAuditLog(actor_id=actor_id, action=data.action, target=data.target, details=data.details)
        db.add(row)
        db.flush()
        db.refresh(row)
    return AuditOut.model_validate(row)


def recent(db: DB, k: int = 50) -> list[AuditOut]:
    rows = db.scalars(select(AdminAuditLog).order_by(AdminAuditLog.created_at.desc(), AdminAuditLog.id.desc()).limit(k))
    return [AuditOut.model_validate(r) for r in rows]
