from datetime import datetime

from pydantic import BaseModel

from app_api.models.enums import FeedbackKind
from app_api.schemas.common import Out


class FeedbackIn(BaseModel):
    kind: FeedbackKind


class FeedbackOut(Out):
    item_id: str
    kind: FeedbackKind
    created_at: datetime
