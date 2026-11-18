"""Shared request-time state and helpers for the routers (filled by the lifespan in serve/app.py)."""
from typing import Literal

from fastapi import HTTPException

from serve.product import Product

# Optional filter on lists (data/item_types.py). None = every item (the trained pipeline's output as is).
ItemType = Literal["game", "accessory", "other"]

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


EXCLUDE_MAX = 500


def exclude_ids(s: str | None) -> list[str]:
    """`exclude` query param: comma-separated item ids the user marked "Not interested" (P1 phase G)."""
    out = ids(s or "")
    if len(out) > EXCLUDE_MAX:
        raise HTTPException(status_code=422, detail=f"exclude: at most {EXCLUDE_MAX} item ids")
    return out


def require_admin(mgr, token):
    if mgr.cfg.admin_token and token != mgr.cfg.admin_token:
        raise HTTPException(status_code=403, detail="bad admin token")
