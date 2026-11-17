"""/library: onboarding picks, the user's recent games (= the history the engine gets), stats."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DB

from app_api.db.session import get_autocommit_db, get_db
from app_api.deps import current_session, require_service
from app_api.schemas.auth import SessionOut
from app_api.schemas.library import LibraryItemOut, LibraryStatsOut, PicksIn
from app_api.services import library_service

router = APIRouter(prefix="/library", tags=["library"], dependencies=[Depends(require_service)])


@router.post("/picks")
def picks(body: PicksIn, s: SessionOut = Depends(current_session), db: DB = Depends(get_db)):
    return {"added": library_service.add_picks(db, s.user.id, body)}


@router.get("", response_model=list[LibraryItemOut])
def recent(k: int = Query(20, ge=1, le=200), games_only: bool = True,
           s: SessionOut = Depends(current_session), db: DB = Depends(get_autocommit_db)):
    return library_service.recent(db, s.user.id, k, games_only)


@router.get("/stats", response_model=LibraryStatsOut)
def stats(s: SessionOut = Depends(current_session), db: DB = Depends(get_autocommit_db)):
    return library_service.stats(db, s.user.id)
