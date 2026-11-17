from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app_api.models.enums import ItemType

DESCRIPTION_MAX = 2000      # cards and the game page need far less; the full text stays in the gold data


class GameIn(BaseModel):
    """One catalog row as the loader sends it (scripts/load_catalog.py)."""
    item_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    title: Annotated[str, StringConstraints(max_length=1000)]
    title_family: Annotated[str, StringConstraints(max_length=1000)]
    item_type: ItemType
    platforms: list[str] = Field(default_factory=list)
    publisher: Annotated[str, StringConstraints(max_length=300)] | None = None
    description: Annotated[str, StringConstraints(max_length=DESCRIPTION_MAX)] | None = None
    listed_on: date | None = None


class CatalogLoadOut(BaseModel):
    rows: int
    inserted: int
    updated: int
    unchanged: int
    deactivated: int
