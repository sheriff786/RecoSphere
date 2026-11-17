"""Python enums for the 5 Postgres enum types, and the column types that map them."""
import enum

from sqlalchemy import Enum as SAEnum
from sqlalchemy.types import UserDefinedType


class UserKind(str, enum.Enum):
    registered = "registered"
    demo = "demo"


class UserRole(str, enum.Enum):
    player = "player"
    admin = "admin"


class ItemType(str, enum.Enum):
    game = "game"
    dlc = "dlc"
    accessory = "accessory"
    console = "console"
    other = "other"


class LibrarySource(str, enum.Enum):
    onboarding = "onboarding"
    play = "play"


class FeedbackKind(str, enum.Enum):
    like = "like"
    dislike = "dislike"
    not_interested = "not_interested"


def pg_enum(py_enum: type[enum.Enum], name: str) -> SAEnum:
    """Native Postgres enum `name`, stored by VALUE ('not_interested'), not by the Python member name."""
    return SAEnum(py_enum, name=name, values_callable=lambda e: [m.value for m in e], validate_strings=True)


class CIText(UserDefinedType):
    """Postgres `citext` (extension): case-insensitive text, used for email."""
    cache_ok = True

    def get_col_spec(self, **kw) -> str:
        return "CITEXT"
