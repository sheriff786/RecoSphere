"""app_api tests run against a real Postgres given by TEST_DATABASE_URL (CI: a Postgres 16 service container).
Without it they are skipped. Nothing outlives a test run:

- `conn`: one connection + one transaction per test, rolled back (schema-building tests use it).
- `db`: a SQLAlchemy Session for service tests. The app schema is built ONCE per run inside an outer
  transaction (in its own schema `rs_app`, from 001_init.sql in a single round trip, which matters on a
  remote Neon), each test runs in a SAVEPOINT that is rolled back, and the outer transaction is rolled back
  at the end. Services call db.commit(); with join_transaction_mode="create_savepoint" that only releases
  a savepoint, so the test still sees and then discards everything.
"""
import os
import re
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from app_api.config import sqlalchemy_url
from app_api.db.session import make_engine
from app_api.models import Game, ItemType

URL = os.environ.get("TEST_DATABASE_URL")
SQL = (Path(__file__).resolve().parents[2] / "web" / "db" / "migrations" / "001_init.sql").read_text(encoding="utf-8")
SCHEMA_SQL = re.sub(r"(?im)^\s*CREATE EXTENSION.*$", "", SQL)

GAMES = [  # a tiny catalog for service tests
    dict(item_id="G-ZELDA", title="The Legend of Zelda: Breath of the Wild - Nintendo Switch",
         title_family="legend of zelda breath of the wild", item_type=ItemType.game, platforms=["Switch"]),
    dict(item_id="G-ZELDA-WIIU", title="The Legend of Zelda: Breath of the Wild - Wii U",
         title_family="legend of zelda breath of the wild", item_type=ItemType.game, platforms=["Wii U"]),
    dict(item_id="G-MARIO", title="Super Mario 3D All-Stars", title_family="super mario 3d all stars", item_type=ItemType.game),
    dict(item_id="G-POKEMON", title="Pokemon Sword", title_family="pokemon sword", item_type=ItemType.game),
    dict(item_id="G-DLC", title="Pokemon Sword Expansion Pass", title_family="pokemon sword expansion pass", item_type=ItemType.dlc),
    dict(item_id="A-PAD", title="Pro Controller", title_family="pro controller", item_type=ItemType.accessory),
    dict(item_id="G-OLD", title="Delisted Game", title_family="delisted game", item_type=ItemType.game, is_active=False),
]


@pytest.fixture(scope="session")
def engine():
    if not URL:
        pytest.skip("set TEST_DATABASE_URL to a Postgres database")
    eng = make_engine(sqlalchemy_url(URL), pool_size=3)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine):
    """One connection, one transaction, always rolled back."""
    with engine.connect() as c:
        tx = c.begin()
        try:
            yield c
        finally:
            tx.rollback()


@pytest.fixture(scope="session")
def app_conn(engine):
    """The app schema (rs_app) + seed games, built once, inside a transaction rolled back at the end."""
    with engine.connect() as c:
        tx = c.begin()
        try:
            c.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS citext")
            c.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            c.exec_driver_sql("CREATE SCHEMA rs_app")
            c.exec_driver_sql("SET LOCAL search_path TO rs_app, public")
            c.connection.driver_connection.execute(SCHEMA_SQL)       # one round trip for the whole schema
            with Session(bind=c) as s:
                s.add_all([Game(**g) for g in GAMES])
                s.flush()
            yield c
        finally:
            tx.rollback()


@pytest.fixture
def db(app_conn):
    nested = app_conn.begin_nested()
    s = Session(bind=app_conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield s
    finally:
        s.close()
        if nested.is_active:
            nested.rollback()


@pytest.fixture
def statements(app_conn):
    """Count the SQL statements a service call sends (a proxy for round trips; BEGIN/COMMIT not included)."""
    seen: list[str] = []

    def count(conn, cursor, statement, *a):
        if not statement.lstrip().upper().startswith(("SAVEPOINT", "RELEASE", "ROLLBACK")):
            seen.append(statement)
    event.listen(app_conn, "before_cursor_execute", count)
    yield seen
    event.remove(app_conn, "before_cursor_execute", count)
