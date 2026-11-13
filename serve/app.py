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

Run locally (in-memory store):   uvicorn serve.app:app --port 8000
Run with Redis:                  REDIS_URL=redis://localhost:6379/0 uvicorn serve.app:app --port 8000
"""
import logging
import threading
import time
from contextlib import asynccontextmanager

import os

from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel

from serve import metrics as M
from serve.config import ServeConfig
from serve.experiments import ExperimentManager, ExposureLogger
from serve.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recosphere")


class ModelManager:
    """Owns the live Recommender. `current` is swapped with a single assignment (atomic in CPython)."""

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.current = Recommender(cfg)
        M.set_model(None, None, self.current.model_run, self.current.emb_version)
        self.experiments = ExperimentManager(cfg, Recommender, self.current.store)
        self._lock = threading.Lock()                 # one reload at a time per worker
        self._stop = threading.Event()
        self.reloads, self.last_error, self.last_check = 0, None, None
        self.failed_target = None                     # don't retry a broken model every tick
        self._thread = None
        if cfg.watch_seconds > 0:
            self._thread = threading.Thread(target=self._watch, name="model-watch", daemon=True)
            self._thread.start()

    def wanted_dir(self):
        return self.cfg.models                        # re-reads production.json

    def reload(self, reason: str, force: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"status": "busy", "detail": "a reload is already running"}
        try:
            target = self.wanted_dir()
            if not force and target == self.current.model_dir:
                return {"status": "unchanged", "model_run": self.current.model_run}
            if not force and target == self.failed_target:
                return {"status": "skipped", "detail": f"{target.name} failed before; use force=true to retry",
                        "model_run": self.current.model_run}
            old = self.current
            log.info("model reload (%s): %s -> %s", reason, old.model_dir.name, target.name)
            t0 = time.perf_counter()
            try:
                new = Recommender(self.cfg, store=old.store)          # old keeps serving meanwhile
                new.smoke_test()
            except Exception as e:                                    # keep serving the old model
                self.last_error = f"{target.name}: {type(e).__name__}: {e}"
                self.failed_target = target
                log.exception("model reload FAILED, still serving %s", old.model_run)
                M.RELOADS.labels("failed").inc()
                return {"status": "failed", "error": self.last_error, "model_run": old.model_run}
            self.current = new                                        # <- the swap
            self.reloads += 1
            M.RELOADS.labels("swapped").inc()
            M.set_model(old.model_run, old.emb_version, new.model_run, new.emb_version)
            self.last_error, self.failed_target = None, None
            secs = time.perf_counter() - t0
            log.info("model swapped to %s in %.1fs", new.model_run, secs)
            return {"status": "swapped", "from": old.model_run, "to": new.model_run, "seconds": round(secs, 1)}
        finally:
            self._lock.release()

    def _watch(self):
        while not self._stop.wait(self.cfg.watch_seconds):
            try:
                self.last_check = time.time()
                want = self.wanted_dir()
                if want == self.current.model_dir and self.failed_target is not None:
                    self.failed_target, self.last_error = None, None   # pointer rolled back: issue resolved
                elif want not in (self.current.model_dir, self.failed_target):
                    self.reload("production.json changed")
                self.experiments.refresh()                                # experiment config edited?
            except Exception:                                         # a bad pointer must not kill the thread
                log.exception("model watch error")

    def stop(self):
        self._stop.set()

    def info(self) -> dict:
        return {"reloads": self.reloads, "last_reload_error": self.last_error, "watch_seconds": self.cfg.watch_seconds,
                **self.experiments.info()}


STATE: dict = {}


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


class Feedback(BaseModel):
    request_id: str
    user_id: str
    item_id: str
    action: str = "click"


class Event(BaseModel):
    user_id: str
    item_id: str
    ts: int | None = None
    rating: float | None = None


@app.get("/health")
def health():
    mgr = STATE["mgr"]
    return mgr.current.info() | mgr.info()


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=50), rerank: bool = True, debug: bool = False,
              force_arm: str | None = None):
    mgr = STATE["mgr"]
    t0 = time.perf_counter()
    try:
        rec, exp, arm, forced = mgr.experiments.route(user_id, mgr.current, force_arm)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown arm {force_arm}")
    try:                                                       # `rec` = one object for the whole request
        out = rec.recommend(user_id, k=k, rerank=rerank)
    except Exception:                                          # never fail the page: log + popular fallback
        log.exception("recommend failed for %s", user_id)
        M.ERRORS.labels("recommend").inc()
        out = {"user_id": user_id, "source": "fallback_error",
               "items": [{"item_id": rec.item_ids[i], "title": rec.titles[i], "is_new": bool(rec.items.is_new[i]),
                          "score": None} for i in rec.fallback[:k]], "timings_ms": {}}
    secs = time.perf_counter() - t0
    M.observe_recommend(out, arm, rec.model_run, secs)
    rid = ExposureLogger.new_request_id()
    out.update(request_id=rid, experiment=exp, arm=arm, model_run=rec.model_run)
    STATE["exposures"].log(request_id=rid, experiment=exp, arm=arm, forced=forced, user_id=user_id,
                           model_run=rec.model_run, source=out["source"], k=k,
                           items=[x["item_id"] for x in out["items"]], n_new=sum(bool(x["is_new"]) for x in out["items"]),
                           latency_ms=round(secs * 1e3, 2))
    if not debug:
        out.pop("timings_ms", None)
    return out


@app.post("/feedback")
def feedback(f: Feedback):
    """Client-side outcome (click/purchase) for a served list. Arm is looked up again from the user id
    (assignment is deterministic), so the client cannot send the wrong arm."""
    mgr = STATE["mgr"]
    _, exp, arm, _ = mgr.experiments.route(f.user_id, mgr.current)
    M.FEEDBACK.labels(arm, f.action).inc()
    STATE["clicks"].log(request_id=f.request_id, experiment=exp, arm=arm, user_id=f.user_id,
                        item_id=f.item_id, action=f.action)
    return {"ok": True}


@app.post("/event")
def event(e: Event):
    try:
        r = STATE["mgr"].current.add_event(e.user_id, e.item_id, e.ts, e.rating)
        M.EVENTS.labels(str(r.get("recorded"))).inc()
        return r
    except KeyError:
        M.ERRORS.labels("unknown_item").inc()
        raise HTTPException(status_code=404, detail=f"unknown item_id {e.item_id}")


@app.post("/admin/reload")
def admin_reload(force: bool = False, x_admin_token: str | None = Header(default=None)):
    mgr = STATE["mgr"]
    if mgr.cfg.admin_token and x_admin_token != mgr.cfg.admin_token:
        raise HTTPException(status_code=403, detail="bad admin token")
    return {"model": mgr.reload("admin request", force=force), "experiment": mgr.experiments.refresh(force=True)}


@app.get("/metrics")
def metrics():
    body, ctype = M.render()
    return Response(content=body, media_type=ctype)
