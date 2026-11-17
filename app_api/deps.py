"""FastAPI dependencies shared by the routers: who is calling (the web app's service token) and for which user
(the login session id the web app keeps in its signed cookie)."""
import hmac
import uuid

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session as DB

from app_api.config import get_settings
from app_api.db.session import get_autocommit_db
from app_api.models import UserRole
from app_api.schemas.auth import SessionOut
from app_api.services import auth_service


def require_service(x_service_token: str | None = Header(default=None)) -> None:
    """Only the web app (server side) may call app_api. Constant-time comparison; fails closed when unset."""
    expected = get_settings().service_token
    if not expected or not x_service_token or not hmac.compare_digest(x_service_token, expected):
        raise HTTPException(status_code=401, detail="service token required")


def current_session(x_session_id: uuid.UUID | None = Header(default=None),
                    db: DB = Depends(get_autocommit_db)) -> SessionOut:
    """The caller's login session (sessions row + user, one autocommit query). 401 if missing, revoked, expired."""
    if x_session_id is None:
        raise HTTPException(status_code=401, detail="session required")
    s = auth_service.current(db, x_session_id)
    if s is None:
        raise HTTPException(status_code=401, detail="session expired or revoked")
    return s


def require_admin(s: SessionOut = Depends(current_session)) -> SessionOut:
    if s.user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="admin only")
    return s
