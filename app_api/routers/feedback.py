"""/feedback: like / dislike / not_interested per game; the ids every row must hide."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DB

from app_api.db.session import get_autocommit_db
from app_api.deps import current_session, require_service
from app_api.schemas.auth import SessionOut
from app_api.schemas.common import ItemId
from app_api.schemas.feedback import FeedbackIn, FeedbackOut
from app_api.services import feedback_service

router = APIRouter(prefix="/feedback", tags=["feedback"], dependencies=[Depends(require_service)])


@router.put("/{item_id}", response_model=FeedbackOut)
def set_feedback(item_id: ItemId, body: FeedbackIn, s: SessionOut = Depends(current_session),
                 db: DB = Depends(get_autocommit_db)):         # one upsert statement
    return feedback_service.set_feedback(db, s.user.id, item_id, body)


@router.delete("/{item_id}")
def clear(item_id: ItemId, s: SessionOut = Depends(current_session), db: DB = Depends(get_autocommit_db)):
    return {"cleared": feedback_service.clear(db, s.user.id, item_id)}


@router.get("/hidden")
def hidden(s: SessionOut = Depends(current_session), db: DB = Depends(get_autocommit_db)):
    return {"item_ids": feedback_service.hidden_item_ids(db, s.user.id)}
