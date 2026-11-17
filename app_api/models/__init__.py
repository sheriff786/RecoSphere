"""Model classes: one SQLAlchemy class per table. Importing this package registers every table on
Base.metadata (Alembic and the drift test rely on that)."""
from app_api.models.audit import AdminAuditLog
from app_api.models.enums import FeedbackKind, ItemType, LibrarySource, UserKind, UserRole
from app_api.models.feedback import ItemFeedback
from app_api.models.game import Game
from app_api.models.library import LibraryItem
from app_api.models.play import PlaySession
from app_api.models.session import Session
from app_api.models.user import User

__all__ = ["AdminAuditLog", "FeedbackKind", "Game", "ItemFeedback", "ItemType", "LibraryItem", "LibrarySource",
           "PlaySession", "Session", "User", "UserKind", "UserRole"]
