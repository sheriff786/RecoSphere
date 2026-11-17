"""Python DB connect: one SQLAlchemy engine (connection pool) per process, one Session per request.

Three settings matter for Neon:
- no pre-ping, recycle after 240 s instead (see make_engine: one round trip less per request).
- prepare_threshold=None: psycopg 3 switches to server-side prepared statements after a few runs of the
  same query; through Neon's -pooler host (PgBouncer, transaction mode) the next transaction can land
  on another server connection where that statement does not exist. Off.
- sslrootcert: with sslmode=verify-full libpq needs a CA bundle; certifi's works on Windows, Linux and
  in Docker alike (libpq's "system" store is not found on Windows).
"""
from collections.abc import Iterator
from functools import lru_cache

import certifi
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app_api.config import get_settings


def connect_args(url: str) -> dict:
    args: dict = {"prepare_threshold": None}
    if "sslmode=verify" in url and "sslrootcert=" not in url:
        args["sslrootcert"] = certifi.where()
    return args


def make_engine(url: str, **kw) -> Engine:
    # No pool_pre_ping: measured on 2026-11-17 (laptop -> Neon us-east-1, warm pool, median of 6), it adds a
    # full round trip to EVERY request: first statement 813 ms with it, 532 ms without (one round trip is
    # about 266 ms). Instead pool_recycle=240 s: Neon's free tier suspends after 5 min idle and drops its
    # connections; any connection idle that long is older than 240 s, so it is replaced before use.
    kw.setdefault("pool_pre_ping", False)
    kw.setdefault("pool_recycle", 240)
    return create_engine(url, connect_args=connect_args(url), **kw)


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    return make_engine(s.app_url, pool_size=s.pool_size, max_overflow=s.max_overflow)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: a Session for one request, always closed (its connection goes back to the pool)."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
