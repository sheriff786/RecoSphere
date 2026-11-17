"""app_api entry point:  uvicorn app_api.main:app --host 127.0.0.1 --port 8100   (make app-api)

Every route except /health needs the web app's X-Service-Token; user routes also X-Session-Id. Service
errors become HTTP: NotFound 404, Conflict 409, Invalid 422, Unauthorized 401. Endpoints: docs/API.md §3.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app_api.db.session import get_engine
from app_api.routers import admin, auth, feedback, health, library, plays
from app_api.services.errors import Conflict, Invalid, NotFound, ServiceError, Unauthorized

STATUS = {NotFound: 404, Conflict: 409, Invalid: 422, Unauthorized: 401}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if get_engine.cache_info().currsize:   # close pooled connections, if the pool was ever created
        get_engine().dispose()


app = FastAPI(title="RecoSphere app API", version="0.2.0", lifespan=lifespan,
              description="User-owned data (accounts, library, plays, feedback, admin audit) on Postgres.")


@app.exception_handler(ServiceError)
def service_error(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=STATUS.get(type(exc), 400), content={"detail": exc.message, "code": exc.code})


for r in (health, auth, library, plays, feedback, admin):
    app.include_router(r.router)
