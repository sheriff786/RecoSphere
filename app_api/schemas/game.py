from datetime import date

from app_api.models.enums import ItemType
from app_api.schemas.common import Out


class GameOut(Out):
    item_id: str
    title: str
    title_family: str
    item_type: ItemType
    platforms: list[str]
    publisher: str | None
    listed_on: date | None
    is_active: bool
