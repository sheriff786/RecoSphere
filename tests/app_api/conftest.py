"""app_api tests run against a real Postgres given by TEST_DATABASE_URL (CI: a Postgres 16 service container).
Without it they are skipped. Every test works inside a transaction that is rolled back, so even a shared
database keeps no rows, tables or schemas from a test run."""
import os

import pytest

from app_api.config import sqlalchemy_url
from app_api.db.session import make_engine

URL = os.environ.get("TEST_DATABASE_URL")
requires_db = pytest.mark.skipif(not URL, reason="set TEST_DATABASE_URL to a Postgres database")


@pytest.fixture(scope="session")
def engine():
    if not URL:
        pytest.skip("set TEST_DATABASE_URL to a Postgres database")
    eng = make_engine(sqlalchemy_url(URL), pool_size=2)
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
