"""POST /event (play/rating -> user history) and POST /feedback (click log = A/B outcome)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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


class HistoryItem(BaseModel):
    item_id: str
    ts: int | None = None
    rating: float | None = None


class History(BaseModel):
    items: list[HistoryItem] = Field(max_length=500)


@router.post("/users/{user_id}/history")
def replay_history(user_id: str, h: History):
    """Bulk /event, oldest first: the app replays a user's saved library after an engine restart (Redis may be
    empty). Same dedup as /event, so replaying twice records nothing new. Unknown items are skipped and listed."""
    r = STATE["mgr"].current.add_events(user_id, [(i.item_id, i.ts, i.rating) for i in h.items])
    M.EVENTS.labels("True").inc(r["recorded"])
    M.EVENTS.labels("False").inc(r["already_seen"])
    if r["unknown"]:
        M.ERRORS.labels("unknown_item").inc(len(r["unknown"]))
    return r
