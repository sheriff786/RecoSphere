"""/auth: signup, login, demo login, logout, me. The web app keeps the returned session_id in its signed cookie
(JWT claim `sid`) and sends it back as X-Session-Id."""
import uuid

from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.orm import Session as DB

from app_api.db.session import get_db
from app_api.deps import current_session, require_service
from app_api.schemas.auth import DemoLoginIn, LoginIn, SessionOut, SignupIn
from app_api.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(require_service)])


@router.post("/signup", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def signup(body: SignupIn, db: DB = Depends(get_db), x_client_user_agent: str | None = Header(default=None)):
    return auth_service.signup(db, body, user_agent=x_client_user_agent)


@router.post("/login", response_model=SessionOut)
def login(body: LoginIn, db: DB = Depends(get_db), x_client_user_agent: str | None = Header(default=None)):
    return auth_service.login(db, body, user_agent=x_client_user_agent)


@router.post("/demo", response_model=SessionOut)
def demo(body: DemoLoginIn, db: DB = Depends(get_db), x_client_user_agent: str | None = Header(default=None)):
    return auth_service.demo_login(db, body, user_agent=x_client_user_agent)


@router.post("/logout")
def logout(x_session_id: uuid.UUID = Header(), db: DB = Depends(get_db)):
    """Revoke this login session. Idempotent: an unknown or already revoked id answers revoked=false."""
    return {"revoked": auth_service.logout(db, x_session_id)}


@router.get("/me", response_model=SessionOut)
def me(s: SessionOut = Depends(current_session)):
    return s
