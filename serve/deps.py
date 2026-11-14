"""Shared request-time state and helpers for the routers (filled by the lifespan in serve/app.py)."""
from fastapi import HTTPException

from serve.product import Product

# mgr (ModelManager), exposures / clicks (ExposureLogger). Set at startup, cleared at shutdown.
STATE: dict = {}


def product(rec=None) -> Product:
    """One Product view per loaded model (rebuilt automatically after a hot swap)."""
    rec = rec or STATE["mgr"].current
    if getattr(rec, "_product", None) is None:
        rec._product = Product(rec)
    return rec._product


def ids(s: str) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def require_admin(mgr, token):
    if mgr.cfg.admin_token and token != mgr.cfg.admin_token:
        raise HTTPException(status_code=403, detail="bad admin token")
