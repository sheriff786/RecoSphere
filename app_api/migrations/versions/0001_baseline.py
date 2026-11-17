"""baseline: the schema of web/db/migrations/001_init.sql (DB_SCHEMA.md), run as is

Neon already has this schema (applied 2026-11-15 by web/db/migrate.mjs): there it is marked as done with
`alembic stamp 0001_baseline`, not run again. A new, empty database gets it with `alembic upgrade head`.
tests/app_api/test_models.py proves the models build exactly this schema.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-17
"""
from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0001_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_FILE = Path(__file__).resolve().parents[3] / "web" / "db" / "migrations" / "001_init.sql"


def upgrade() -> None:
    # the raw psycopg connection runs a multi-statement script when no parameters are passed
    op.get_bind().connection.driver_connection.execute(SQL_FILE.read_text(encoding="utf-8"))


def downgrade() -> None:
    raise NotImplementedError("forward-only migrations: write a new revision instead")
