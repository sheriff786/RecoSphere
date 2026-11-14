"""Product API for the web app: /catalog/* and /users/* (serve/product.py does the work)."""
from fastapi import APIRouter, HTTPException, Query

from serve import deps
from serve.deps import product

router = APIRouter()


@router.get("/catalog/items")
def catalog_items(ids: str):
    return {"items": product().items(deps.ids(ids))}


@router.get("/catalog/popular")
def catalog_popular(k: int = Query(24, ge=1, le=100), offset: int = Query(0, ge=0), new_only: bool = False):
    return {"items": product().popular(k, offset, new_only)}


@router.get("/catalog/search")
def catalog_search(q: str, k: int = Query(24, ge=1, le=100)):
    return {"query": q, "items": product().search(q, k)}


@router.get("/catalog/similar/{item_id}")
def catalog_similar(item_id: str, k: int = Query(12, ge=1, le=50), user_id: str | None = None):
    p = product()
    if item_id not in p.rec.iid_of:
        raise HTTPException(status_code=404, detail=f"unknown item_id {item_id}")
    seen = [x["item_id"] for x in p.history(user_id, k=10_000)] if user_id else []
    return {"item_id": item_id, "items": p.similar(item_id, k, exclude=seen)}


@router.get("/users/demo")
def users_demo(k: int = Query(12, ge=1, le=50)):
    return {"users": product().demo_users(k)}


@router.get("/users/{user_id}/history")
def user_history(user_id: str, k: int = Query(20, ge=1, le=200)):
    return {"user_id": user_id, "items": product().history(user_id, k)}
