"""/admin: the audit log of admin actions. Admins only (users.role = 'admin')."""
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session as DB

from app_api.db.session import get_autocommit_db, get_db
from app_api.deps import require_admin, require_service
from app_api.schemas.audit import AuditIn, AuditOut
from app_api.schemas.auth import SessionOut
from app_api.services import audit_service

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_service)])


@router.post("/audit", response_model=AuditOut, status_code=status.HTTP_201_CREATED)
def record(body: AuditIn, s: SessionOut = Depends(require_admin), db: DB = Depends(get_db)):
    return audit_service.record(db, s.user.id, body)


@router.get("/audit", response_model=list[AuditOut])
def recent(k: int = Query(50, ge=1, le=500), _: SessionOut = Depends(require_admin),
           db: DB = Depends(get_autocommit_db)):
    return audit_service.recent(db, k)
