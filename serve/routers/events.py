"""POST /event (play/rating -> user history) and POST /feedback (click log = A/B outcome)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from serve import metrics as M
from serve.deps import STATE

router = APIRouter()


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


@router.post("/feedback")
def feedback(f: Feedback):
    """Client-side outcome (click/purchase) for a served list. Arm is looked up again from the user id
    (assignment is deterministic), so the client cannot send the wrong arm."""
    mgr = STATE["mgr"]
    _, exp, arm, _ = mgr.experiments.route(f.user_id, mgr.current)
    M.FEEDBACK.labels(arm, f.action).inc()
    STATE["clicks"].log(request_id=f.request_id, experiment=exp, arm=arm, user_id=f.user_id,
                        item_id=f.item_id, action=f.action)
    return {"ok": True}


@router.post("/event")
def event(e: Event):
    try:
        r = STATE["mgr"].current.add_event(e.user_id, e.item_id, e.ts, e.rating)
        M.EVENTS.labels(str(r.get("recorded"))).inc()
        return r
    except KeyError:
        M.ERRORS.labels("unknown_item").inc()
        raise HTTPException(status_code=404, detail=f"unknown item_id {e.item_id}")
