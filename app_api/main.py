"""app_api entry point:  uvicorn app_api.main:app --host 127.0.0.1 --port 8100   (make app-api)

Phase A (docs/platform/P1_ROADMAP.md): DB connection, models, Alembic and /health. Auth, library, plays,
feedback and admin routers arrive in phases B-G.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app_api.db.session import get_engine
from app_api.routers import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if get_engine.cache_info().currsize:   # close pooled connections, if the pool was ever created
        get_engine().dispose()


app = FastAPI(title="RecoSphere app API", version="0.1.0", lifespan=lifespan,
              description="User-owned data (accounts, library, plays, feedback, admin audit) on Postgres.")
app.include_router(health.router)
