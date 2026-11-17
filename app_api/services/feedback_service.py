"""Explicit feedback: one current value per (user, game). dislike / not_interested hide a game from rows."""
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DB

from app_api.models import FeedbackKind, ItemFeedback
from app_api.schemas.feedback import FeedbackIn, FeedbackOut
from app_api.services.errors import NotFound
from app_api.services.tx import transaction

HIDES = (FeedbackKind.dislike, FeedbackKind.not_interested)


def set_feedback(db: DB, user_id: uuid.UUID, item_id: str, data: FeedbackIn) -> FeedbackOut:
    """Set or change the user's feedback on a game (changing your mind overwrites; created_at = when set)."""
    try:
        with transaction(db):
            row = db.scalar(pg_insert(ItemFeedback)
                            .values(user_id=user_id, item_id=item_id, kind=data.kind)
                            .on_conflict_do_update(index_elements=["user_id", "item_id"],
                                                   set_={"kind": data.kind, "created_at": func.now()})
                            .returning(ItemFeedback))
    except IntegrityError as e:
        if "item_feedback_item_id_fkey" in str(e.orig):
            raise NotFound(f"unknown item {item_id}") from None
        raise
    return FeedbackOut.model_validate(row)


def clear(db: DB, user_id: uuid.UUID, item_id: str) -> bool:
    with transaction(db):
        return db.execute(delete(ItemFeedback).where(ItemFeedback.user_id == user_id,
                                                     ItemFeedback.item_id == item_id)).rowcount == 1


def hidden_item_ids(db: DB, user_id: uuid.UUID) -> list[str]:
    """Games every row must drop for this user."""
    return list(db.scalars(select(ItemFeedback.item_id).where(ItemFeedback.user_id == user_id,
                                                               ItemFeedback.kind.in_(HIDES))
                           .order_by(ItemFeedback.item_id)))
