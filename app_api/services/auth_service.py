"""Accounts and login sessions: signup, login, demo login, logout, session -> user.

A login session is a `sessions` row; its id goes into the cookie's JWT as `sid` (phase D), so logging out
or banning a user really ends access (a stateless JWT cannot be revoked).
"""
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DB
from sqlalchemy.orm import joinedload

from app_api.models import Session, User, UserKind
from app_api.schemas.auth import DemoLoginIn, LoginIn, SessionOut, SignupIn
from app_api.schemas.user import UserOut
from app_api.services.errors import Conflict, Unauthorized
from app_api.services.tx import transaction

SESSION_DAYS = 30
BCRYPT_ROUNDS = 12          # tests lower it; 12 is about 0.25 s per hash on a laptop core
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalizer", bcrypt.gensalt(rounds=4))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, password_hash: str | None) -> bool:
    pw = password.encode("utf-8")
    if password_hash is None or len(pw) > 72:          # bcrypt >= 5 raises on > 72 bytes: just "wrong"
        bcrypt.checkpw(b"x", _DUMMY_HASH)              # same work either way: no timing hint
        return False
    return bcrypt.checkpw(pw, password_hash.encode("ascii"))


def _new_session(db: DB, user: User, user_agent: str | None) -> Session:
    s = Session(user=user, expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS),
                user_agent=(user_agent or "")[:300] or None)
    db.add(s)
    return s


def _out(s: Session) -> SessionOut:
    u = UserOut.model_validate(s.user)
    return SessionOut(session_id=s.id, expires_at=s.expires_at, user=u, needs_onboarding=u.onboarded_at is None)


def signup(db: DB, data: SignupIn, user_agent: str | None = None) -> SessionOut:
    name = data.display_name or data.email.split("@")[0][:40]
    try:
        with transaction(db):
            user = User(kind=UserKind.registered, email=data.email, password_hash=hash_password(data.password),
                        display_name=name)
            db.add(user)
            s = _new_session(db, user, user_agent)
            db.flush()
    except IntegrityError as e:
        if "users_email_key" in str(e.orig):
            raise Conflict("an account with this email exists") from None
        raise
    return _out(s)


def login(db: DB, data: LoginIn, user_agent: str | None = None) -> SessionOut:
    with transaction(db):
        user = db.scalar(select(User).where(User.email == data.email, User.kind == UserKind.registered))
        if not verify_password(data.password, user.password_hash if user else None):
            raise Unauthorized("wrong email or password")
        user.last_login_at = func.now()
        s = _new_session(db, user, user_agent)
        db.flush()
        db.refresh(user, ["last_login_at"])
    return _out(s)


def demo_login(db: DB, data: DemoLoginIn, user_agent: str | None = None) -> SessionOut:
    """Sign in as a real dataset player. The CALLER must check that the engine knows this id (the web app does,
    against /users/demo); the row is created on the first visit so the player's in-app plays can be stored."""
    with transaction(db):
        db.execute(pg_insert(User).values(kind=UserKind.demo, dataset_user_id=data.dataset_user_id,
                                          display_name=f"Player {data.dataset_user_id[:8]}",
                                          onboarded_at=func.now())
                   .on_conflict_do_nothing(index_elements=["dataset_user_id"]))
        user = db.scalar(select(User).where(User.dataset_user_id == data.dataset_user_id))
        user.last_login_at = func.now()
        s = _new_session(db, user, user_agent)
        db.flush()
        db.refresh(user, ["last_login_at"])
    return _out(s)


def logout(db: DB, session_id: uuid.UUID) -> bool:
    """Revoke one login session. True if it was active."""
    with transaction(db):
        n = db.execute(update(Session).where(Session.id == session_id, Session.revoked_at.is_(None))
                       .values(revoked_at=func.now())).rowcount
    return n == 1


def logout_everywhere(db: DB, user_id: uuid.UUID) -> int:
    with transaction(db):
        return db.execute(update(Session).where(Session.user_id == user_id, Session.revoked_at.is_(None))
                          .values(revoked_at=func.now())).rowcount


def current(db: DB, session_id: uuid.UUID) -> SessionOut | None:
    """The hot path (every page): active session + its user in ONE query. None = logged out."""
    s = db.scalar(select(Session).options(joinedload(Session.user))
                  .where(Session.id == session_id, Session.revoked_at.is_(None), Session.expires_at > func.now()))
    return _out(s) if s else None
