"""GET /health of app_api, through the real FastAPI app with the request's DB session bound to the test
transaction (so it runs against the test database and leaves nothing behind)."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app_api.db.session import get_db
from app_api.main import app


def test_health_reports_db_and_revision(conn):
    conn.exec_driver_sql("CREATE SCHEMA rs_health")
    conn.exec_driver_sql("SET LOCAL search_path TO rs_health, public")
    conn.exec_driver_sql("CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY)")
    conn.exec_driver_sql("INSERT INTO alembic_version VALUES ('0001_baseline')")
    app.dependency_overrides[get_db] = lambda: Session(bind=conn)
    try:
        r = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    body = r.json()
    assert r.status_code == 200 and body["database"] == "ok" and body["schema_revision"] == "0001_baseline", body
    assert body["db_roundtrip_ms"] >= 0


def test_health_is_503_when_the_database_is_down():
    class Down:
        def execute(self, *_):
            raise OperationalError("SELECT 1", {}, SimpleNamespace())
    app.dependency_overrides[get_db] = lambda: Down()
    try:
        r = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()["status"] == "error", r.json()
