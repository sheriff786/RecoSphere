"""GET /health: is the database reachable, and at which migration revision is it."""
import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.orm import Session

from app_api.db.session import get_db

router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        ms = (time.perf_counter() - t0) * 1e3
        try:
            revision = db.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
        except ProgrammingError:                    # table missing: database never migrated / stamped
            db.rollback()
            revision = None
    except DBAPIError as e:                          # unreachable, auth failed, Neon waking up too slowly ...
        return JSONResponse(status_code=503, content={"status": "error", "database": type(e.orig).__name__})
    return {"status": "ok", "database": "ok", "schema_revision": revision, "db_roundtrip_ms": round(ms, 1)}
