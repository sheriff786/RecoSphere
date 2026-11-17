from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Index, Text, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app_api.db.base import Base
from app_api.models.enums import ItemType, pg_enum


class Game(Base):
    """games: the app's copy of the catalog (scripts/load_catalog.py upserts it per model release)."""
    __tablename__ = "games"
    __table_args__ = (
        Index("games_family", "title_family"),
        Index("games_playable", "item_id", postgresql_where=text("item_type IN ('game', 'dlc') AND is_active")),
        Index("games_title_trgm", "title", postgresql_using="gin", postgresql_ops={"title": "gin_trgm_ops"}),
    )

    item_id: Mapped[str] = mapped_column(Text, primary_key=True)             # ASIN, the id the engine uses
    title: Mapped[str] = mapped_column(Text)
    title_family: Mapped[str] = mapped_column(Text)                          # normalized title: edition dedup key
    item_type: Mapped[ItemType] = mapped_column(pg_enum(ItemType, "item_type"))
    platforms: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    publisher: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    listed_on: Mapped[date | None] = mapped_column(Date)                     # Amazon date first available, NOT release
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    source: Mapped[str] = mapped_column(Text, server_default=text("'amazon_reviews_2023'"))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    @property
    def playable(self) -> bool:
        return self.is_active and self.item_type in (ItemType.game, ItemType.dlc)
