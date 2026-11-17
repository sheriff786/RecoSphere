import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app_api.schemas.common import ItemId, Out

Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class PlayStartIn(BaseModel):
    item_id: ItemId
    client_event_id: uuid.UUID          # made by the browser; a retried request with the same id is a no-op
    request_id: Short | None = None     # engine request_id of the list the game was clicked in (attribution)
    surface: Short | None = None        # e.g. 'home:top_picks', 'game:more_like_this', 'search'


class PlayEndIn(BaseModel):
    client_event_id: uuid.UUID          # the id sent with the start
    seconds: int = Field(ge=0, le=24 * 3600)


class PlayOut(Out):
    play_id: uuid.UUID
    item_id: str
    started_at: datetime
    ended_at: datetime | None
    seconds_played: int | None
    first_time: bool                    # the game was not in the user's library before this play
