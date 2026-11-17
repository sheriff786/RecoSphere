"""Admin tasks that are not HTTP endpoints.

    python -m app_api.cli create-admin --email admin@recosphere.dev      # password from $ADMIN_PASSWORD

Creates a registered admin (onboarded), or promotes an existing account to admin (its password is then left
alone unless --reset-password). Uses DATABASE_URL_DIRECT, like Alembic.
"""
import argparse
import os
import sys

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app_api.config import get_settings
from app_api.db.session import make_engine
from app_api.models import User, UserKind, UserRole
from app_api.schemas.auth import SignupIn
from app_api.services.auth_service import hash_password
from app_api.services.tx import transaction


def create_admin(db: Session, email: str, password: str | None, reset_password: bool = False) -> str:
    with transaction(db):
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            if not password:
                raise SystemExit("a new admin needs a password: set ADMIN_PASSWORD")
            SignupIn(email=email, password=password)                     # same rules as a signup
            db.add(User(kind=UserKind.registered, email=email, password_hash=hash_password(password),
                        display_name="Admin", role=UserRole.admin, onboarded_at=func.now()))
            return "created"
        user.role = UserRole.admin
        if reset_password and password:
            SignupIn(email=email, password=password)
            user.password_hash = hash_password(password)
        return "promoted"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app_api.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin")
    c.add_argument("--email", required=True)
    c.add_argument("--reset-password", action="store_true")
    a = ap.parse_args(argv)
    engine = make_engine(get_settings().migrations_url, pool_size=1)
    with Session(engine) as db:
        print(f"{a.email}: {create_admin(db, a.email, os.environ.get('ADMIN_PASSWORD'), a.reset_password)}")
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
