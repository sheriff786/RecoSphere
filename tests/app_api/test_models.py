"""The model classes ARE the schema in 001_init.sql (P1 phase A).

Inside one rolled-back transaction: schema rs_sql is built by running web/db/migrations/001_init.sql, schema
rs_models by Base.metadata.create_all(). Then the Postgres catalog of both is compared: columns (type,
nullability, default, identity), every constraint's definition (PK, FK + ON DELETE, UNIQUE, CHECK), every
index definition (partial, expression, GIN + opclass) and the enum labels. Any difference fails.
A second test runs Alembic's autogenerate comparison against the SQL-built schema: it must find nothing,
or `alembic revision --autogenerate` would later produce changes nobody asked for.
"""
import re
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

import app_api.models  # noqa: F401  (registers every table)
from app_api.db.base import Base, include_name

SQL = (Path(__file__).resolve().parents[2] / "web" / "db" / "migrations" / "001_init.sql").read_text(encoding="utf-8")
SCHEMA_SQL = re.sub(r"(?im)^\s*CREATE EXTENSION.*$", "", SQL)    # extensions are database-wide: created once below


def build_both(conn):
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS citext")
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    conn.exec_driver_sql("CREATE SCHEMA rs_sql")
    conn.exec_driver_sql("CREATE SCHEMA rs_models")
    conn.exec_driver_sql("SET LOCAL search_path TO rs_sql, public")
    conn.exec_driver_sql(SCHEMA_SQL)
    conn.exec_driver_sql("SET LOCAL search_path TO rs_models, public")
    Base.metadata.create_all(conn, checkfirst=False)      # checkfirst would see other schemas' tables


def snapshot(conn, schema: str) -> dict:
    strip = lambda s: None if s is None else re.sub(r"\brs_(sql|models)\.", "", s)
    cols = conn.exec_driver_sql(
        """SELECT table_name, column_name, data_type, udt_name, is_nullable, column_default, is_identity,
                  identity_generation FROM information_schema.columns WHERE table_schema = %s""", (schema,)).all()
    cons = conn.exec_driver_sql(
        """SELECT t.relname, c.conname, c.contype, pg_get_constraintdef(c.oid) FROM pg_constraint c
           JOIN pg_class t ON t.oid = c.conrelid WHERE c.connamespace = %s::regnamespace""", (schema,)).all()
    idx = conn.exec_driver_sql("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s", (schema,)).all()
    enums = conn.exec_driver_sql(
        """SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) FROM pg_type t
           JOIN pg_enum e ON e.enumtypid = t.oid WHERE t.typnamespace = %s::regnamespace GROUP BY t.typname""",
        (schema,)).all()
    return {
        "columns": {(r[0], r[1]): tuple(strip(x) if isinstance(x, str) else x for x in r[2:]) for r in cols},
        "constraints": {(r[0], r[1]): (r[2], strip(r[3])) for r in cons},
        "indexes": {r[0]: strip(r[1]).replace(f" ON {schema}.", " ON ") for r in idx},
        "enums": {r[0]: list(r[1]) for r in enums},
    }


def diff(a: dict, b: dict) -> list[str]:
    out = []
    for kind in a:
        for k in sorted(set(a[kind]) | set(b[kind]), key=str):
            if a[kind].get(k) != b[kind].get(k):
                out.append(f"{kind} {k}:\n    sql    = {a[kind].get(k)}\n    models = {b[kind].get(k)}")
    return out


def test_models_build_exactly_the_sql_schema(conn):
    build_both(conn)
    sql, models = snapshot(conn, "rs_sql"), snapshot(conn, "rs_models")
    assert len(sql["columns"]) > 50 and len(sql["constraints"]) > 20, "snapshot looks empty: test is broken"
    problems = diff(sql, models)
    assert not problems, "models differ from 001_init.sql:\n" + "\n".join(problems)


def test_alembic_autogenerate_sees_no_change(conn):
    build_both(conn)
    conn.exec_driver_sql("SET LOCAL search_path TO rs_sql, public")
    mc = MigrationContext.configure(conn, opts={"compare_type": True, "compare_server_default": True,
                                                "include_name": include_name})
    changes = compare_metadata(mc, Base.metadata)
    assert changes == [], "alembic autogenerate would emit:\n" + "\n".join(map(str, changes))


@pytest.mark.parametrize("table", ["users", "sessions", "games", "library", "play_sessions", "item_feedback", "admin_audit_log"])
def test_every_table_has_a_model(table):
    assert table in Base.metadata.tables
