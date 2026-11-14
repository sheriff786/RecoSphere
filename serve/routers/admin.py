"""Ops + admin: GET /health, GET /metrics, POST /admin/reload, GET /admin/summary, POST /admin/rollback."""
import json

from fastapi import APIRouter, Header, Response

from serve import metrics as M
from serve.deps import STATE, require_admin

router = APIRouter()


@router.get("/health")
def health():
    mgr = STATE["mgr"]
    return mgr.current.info() | mgr.info()


@router.get("/metrics")
def metrics():
    body, ctype = M.render()
    return Response(content=body, media_type=ctype)


@router.post("/admin/reload")
def admin_reload(force: bool = False, x_admin_token: str | None = Header(default=None)):
    mgr = STATE["mgr"]
    require_admin(mgr, x_admin_token)
    return {"model": mgr.reload("admin request", force=force), "experiment": mgr.experiments.refresh(force=True)}


@router.get("/admin/summary")
def admin_summary(x_admin_token: str | None = Header(default=None)):
    mgr = STATE["mgr"]
    require_admin(mgr, x_admin_token)
    hist = []
    hp = mgr.cfg.models_root / "production_history.jsonl"
    if hp.exists():
        hist = [json.loads(x) for x in hp.read_text().splitlines() if x.strip()][-15:][::-1]
    return {"health": mgr.current.info() | mgr.info(), "live": M.snapshot(), "production_history": hist}


@router.post("/admin/rollback")
def admin_rollback(x_admin_token: str | None = Header(default=None)):
    """Point production.json at the previous model; every worker hot-swaps within MODEL_WATCH_SECONDS."""
    mgr = STATE["mgr"]
    require_admin(mgr, x_admin_token)
    from pipeline.config import Paths, load_config
    from pipeline.gate import rollback
    lines = []
    entry = rollback(Paths(load_config(None, [f"category={mgr.cfg.category}"])), log=lines.append)
    return {"rolled_back_to": entry["run_id"], "log": lines, "swap": mgr.reload("admin rollback")}
