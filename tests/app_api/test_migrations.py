"""Alembic (P1 phase A): `upgrade head` on an empty schema builds the model schema, and a later
autogenerate finds nothing to do. Runs inside the rolled-back test transaction (safe on any database)."""
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

import app_api.models  # noqa: F401
from app_api.db.base import Base, include_name

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(conn, version_schema: str | None = None) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["connection"] = conn
    cfg.attributes["version_table_schema"] = version_schema
    return cfg


def test_single_head_is_the_baseline():
    heads = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_heads()
    assert heads == ["0001_baseline"], heads


def test_upgrade_head_on_an_empty_schema_matches_the_models(conn):
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS citext")
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn.exec_driver_sql("CREATE SCHEMA rs_alembic")
    conn.exec_driver_sql("SET LOCAL search_path TO rs_alembic, public")
    command.upgrade(alembic_config(conn, version_schema="rs_alembic"), "head")
    version = conn.exec_driver_sql("SELECT version_num FROM rs_alembic.alembic_version").scalar_one()
    assert version == "0001_baseline"
    tables = set(conn.exec_driver_sql(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'rs_alembic'").scalars())
    assert set(Base.metadata.tables) <= tables, tables
    changes = compare_metadata(MigrationContext.configure(conn, opts={
        "compare_type": True, "compare_server_default": True, "include_name": include_name}), Base.metadata)
    assert changes == [], changes
