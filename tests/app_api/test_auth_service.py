"""auth_service against a real Postgres (see conftest: everything is rolled back)."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app_api.models import Session, User, UserKind
from app_api.schemas.auth import DemoLoginIn, LoginIn, SignupIn
from app_api.services import auth_service as auth
from app_api.services.errors import Conflict, Unauthorized


@pytest.fixture(autouse=True)
def fast_bcrypt(monkeypatch):
    monkeypatch.setattr(auth, "BCRYPT_ROUNDS", 4)


def signup(db, email="Sheriff@Example.com", pw="correct-horse-1"):
    return auth.signup(db, SignupIn(email=email, password=pw), user_agent="pytest")


def test_signup_creates_user_and_session(db):
    out = signup(db)
    assert out.user.kind == UserKind.registered and out.user.display_name == "Sheriff"
    assert out.needs_onboarding and out.expires_at > datetime.now(timezone.utc) + timedelta(days=29)
    stored = db.get(User, out.user.id)
    assert stored.password_hash.startswith("$2b$") and "correct-horse-1" not in stored.password_hash
    assert out.user.engine_user_id == f"app:{out.user.id}"


def test_duplicate_email_in_any_case_is_a_conflict(db):
    signup(db)
    with pytest.raises(Conflict):
        signup(db, email="sheriff@EXAMPLE.com")
    assert db.scalar(select(User).where(User.email == "SHERIFF@example.com")) is not None   # first one kept


def test_login(db):
    first = signup(db)
    with pytest.raises(Unauthorized):
        auth.login(db, LoginIn(email="sheriff@example.com", password="wrong-password"))
    with pytest.raises(Unauthorized):
        auth.login(db, LoginIn(email="nobody@example.com", password="correct-horse-1"))
    with pytest.raises(Unauthorized):                      # > 72 bytes: just wrong, not a crash
        auth.login(db, LoginIn(email="sheriff@example.com", password="é" * 50))
    out = auth.login(db, LoginIn(email="SHERIFF@example.com", password="correct-horse-1"))
    assert out.user.id == first.user.id and out.session_id != first.session_id
    assert db.get(User, out.user.id).last_login_at is not None


def test_current_logout_and_expiry(db, statements):
    out = signup(db)
    statements.clear()
    me = auth.current(db, out.session_id)
    assert me.user.id == out.user.id
    assert len(statements) == 1, statements                # the hot path: session + user in ONE query
    assert auth.logout(db, out.session_id) is True
    assert auth.current(db, out.session_id) is None        # a revoked session is logged out
    assert auth.logout(db, out.session_id) is False
    assert auth.current(db, uuid.uuid4()) is None
    old = Session(user_id=out.user.id, created_at=datetime.now(timezone.utc) - timedelta(days=40),
                  expires_at=datetime.now(timezone.utc) - timedelta(days=10))
    db.add(old)
    db.flush()
    assert auth.current(db, old.id) is None                # an expired session is logged out


def test_logout_everywhere(db):
    a = signup(db)
    b = auth.login(db, LoginIn(email="sheriff@example.com", password="correct-horse-1"))
    assert auth.logout_everywhere(db, a.user.id) == 2
    assert auth.current(db, a.session_id) is None and auth.current(db, b.session_id) is None


def test_demo_login_creates_the_player_once(db):
    one = auth.demo_login(db, DemoLoginIn(dataset_user_id="AEPQB547GDARJWCYCDJO4X2WH5CQ"))
    two = auth.demo_login(db, DemoLoginIn(dataset_user_id="AEPQB547GDARJWCYCDJO4X2WH5CQ"))
    assert one.user.id == two.user.id and one.session_id != two.session_id
    assert one.user.kind == UserKind.demo and not one.needs_onboarding     # real players skip onboarding
    assert one.user.engine_user_id == "AEPQB547GDARJWCYCDJO4X2WH5CQ"
