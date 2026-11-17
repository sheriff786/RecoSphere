"""/plays: start and end a stream. Start runs on the autocommit session: its happy path is ONE statement
(play_service.start_statement), atomic by itself; end writes twice, so it runs in a transaction."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DB

from app_api.db.session import get_autocommit_db, get_db
from app_api.deps import current_session, require_service
from app_api.schemas.auth import SessionOut
from app_api.schemas.play import PlayEndIn, PlayOut, PlayStartIn
from app_api.services import play_service

router = APIRouter(prefix="/plays", tags=["plays"], dependencies=[Depends(require_service)])


@router.post("/start", response_model=PlayOut)
def start(body: PlayStartIn, s: SessionOut = Depends(current_session), db: DB = Depends(get_autocommit_db)):
    return play_service.start(db, s.user.id, body)


@router.post("/end", response_model=PlayOut)
def end(body: PlayEndIn, s: SessionOut = Depends(current_session), db: DB = Depends(get_db)):
    return play_service.end(db, s.user.id, body)
