"""
RecoSphere serving API.

  GET  /health                     model versions, store backend, counts, reload status
  GET  /recommend/{user_id}?k=10   top-k (personalized, or fallback for users without history)
  POST /event                      {"user_id", "item_id", "ts"?, "rating"?} -> updates the user's history
  POST /feedback                   {"request_id", "user_id", "item_id", "action"} -> click log (A/B outcome)
  POST /admin/reload               reload production model + experiment config NOW (header X-Admin-Token
                                   if ADMIN_TOKEN is set)
  GET  /metrics                    Prometheus metrics (Phase 7g)

A/B experiments (Phase 8): experiments/active.yaml (serve/experiments.py). Every /recommend response
carries request_id, experiment, arm, model_run; with EXPOSURE_DIR set, each one is also written to
exposures-<pid>.jsonl and each /feedback to clicks-<pid>.jsonl -> ab/analyze.py joins them.

Zero-downtime model swap (Phase 7d):
  promote/rollback rewrites models/<cat>/production.json  ->  every worker notices within
  MODEL_WATCH_SECONDS (default 10)  ->  loads the new model in a background thread while the OLD one
  keeps serving  ->  smoke test (real requests)  ->  one reference swap. In-flight requests finish on the
  object they started with. If loading or the smoke test fails, the old model stays and /health shows
  the error. The feature store (user state + live events) is shared and NOT reloaded.

Layout (P0): this file only creates the app, runs the lifespan and wires serve/routers/*:
  recommend.py (/recommend, /explain) · events.py (/event, /feedback) · catalog.py (/catalog, /users)
  admin.py (/health, /metrics, /admin/*). Hot swap: serve/model_manager.py. Shared state: serve/deps.py.

Run locally (in-memory store):   uvicorn serve.app:app --port 8000
Run with Redis:                  REDIS_URL=redis://localhost:6379/0 uvicorn serve.app:app --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from serve.config import ServeConfig
from serve.deps import STATE
from serve.experiments import ExposureLogger
from serve.model_manager import ModelManager
from serve.routers import admin, catalog, events, recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recosphere")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["mgr"] = ModelManager(ServeConfig.from_env())        # models + index + feature store
    d = os.environ.get("EXPOSURE_DIR")
    STATE["exposures"], STATE["clicks"] = ExposureLogger(d), ExposureLogger(d)
    if STATE["clicks"].f is not None:                          # separate file for the client half
        STATE["clicks"].f.close()
        STATE["clicks"].f = open(os.path.join(d, f"clicks-{os.getpid()}.jsonl"), "a", buffering=1, encoding="utf-8")
    yield
    STATE["mgr"].stop()
    STATE["exposures"].close(); STATE["clicks"].close()
    STATE.clear()


app = FastAPI(title="RecoSphere", version="0.8.0", lifespan=lifespan)
for r in (admin, recommend, events, catalog):
    app.include_router(r.router)
