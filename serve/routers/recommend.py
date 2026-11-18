"""GET /recommend/{user_id} (A/B routed, exposure-logged) and GET /explain/{user_id}."""
import logging
import time

from fastapi import APIRouter, HTTPException, Query

from serve import deps
from serve import metrics as M
from serve.deps import STATE, ItemType, product
from serve.experiments import ExposureLogger

log = logging.getLogger("recosphere")
router = APIRouter()


@router.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=50), rerank: bool = True, debug: bool = False,
              force_arm: str | None = None, meta: bool = False, item_type: ItemType | None = None,
              exclude: str | None = None):
    mgr = STATE["mgr"]
    ex = deps.exclude_ids(exclude)
    t0 = time.perf_counter()
    try:
        rec, exp, arm, forced = mgr.experiments.route(user_id, mgr.current, force_arm)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown arm {force_arm}")
    try:                                                       # `rec` = one object for the whole request
        out = rec.recommend(user_id, k=k, rerank=rerank, item_type=item_type, exclude=ex)
    except Exception:                                          # never fail the page: log + popular fallback
        log.exception("recommend failed for %s", user_id)
        M.ERRORS.labels("recommend").inc()
        out = {"user_id": user_id, "source": "fallback_error",
               "items": [{"item_id": rec.item_ids[i], "title": rec.titles[i], "is_new": bool(rec.items.is_new[i]),
                          "item_type": str(rec.item_type[i]), "score": None}
                         for i in rec.keep(rec.fallback, item_type, rec.iids(ex))[:k]], "timings_ms": {}}
    secs = time.perf_counter() - t0
    M.observe_recommend(out, arm, rec.model_run, secs)
    rid = ExposureLogger.new_request_id()
    out.update(request_id=rid, experiment=exp, arm=arm, model_run=rec.model_run)
    STATE["exposures"].log(request_id=rid, experiment=exp, arm=arm, forced=forced, user_id=user_id,
                           model_run=rec.model_run, source=out["source"], k=k,
                           items=[x["item_id"] for x in out["items"]], n_new=sum(bool(x["is_new"]) for x in out["items"]),
                           latency_ms=round(secs * 1e3, 2))
    if meta:                                                    # card data for the app (title, platforms, year ...)
        p = product(rec)
        out["items"] = [{**p.items([x["item_id"]])[0], "score": x["score"]} for x in out["items"]]
    if not debug:
        out.pop("timings_ms", None)
    return out


@router.get("/explain/{user_id}")
def explain(user_id: str, items: str | None = None, item_type: ItemType | None = None, exclude: str | None = None):
    mgr = STATE["mgr"]
    rec, _, arm, _ = mgr.experiments.route(user_id, mgr.current)   # explain with the model this user is served
    return {"user_id": user_id, "arm": arm, "model_run": rec.model_run,
            **product(rec).explain(user_id, deps.ids(items) or None, item_type=item_type,
                                   exclude=deps.exclude_ids(exclude))}
