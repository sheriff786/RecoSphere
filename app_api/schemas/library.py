from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app_api.models.enums import LibrarySource
from app_api.schemas.common import ItemId, Out


class PicksIn(BaseModel):
    """Onboarding: at least 3 games, at most 20 (the rest would be ignored by the engine's history anyway)."""
    item_ids: list[ItemId] = Field(min_length=3, max_length=20)

    @field_validator("item_ids")
    @classmethod
    def unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate item_ids")
        return v


class LibraryItemOut(Out):
    item_id: str
    title: str
    source: LibrarySource
    added_at: datetime
    last_played_at: datetime | None
    play_count: int
    seconds_played: int


class LibraryStatsOut(Out):
    games: int
    plays: int
    seconds_played: int
