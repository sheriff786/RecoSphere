"""Card fields parsed from a gold item ("title | Video Games > PC > Games | ... | description"), shared by the
engine (serve/product.py) and the app catalog loader (scripts/load_catalog.py) so both show the same values."""
from __future__ import annotations

from datetime import date, datetime, timezone

PLATFORMS = [("PlayStation 5", "PS5"), ("PlayStation 4", "PS4"), ("PlayStation 3", "PS3"), ("PS Vita", "Vita"),
             ("Xbox Series", "Xbox Series"), ("Xbox One", "Xbox One"), ("Xbox 360", "Xbox 360"),
             ("Nintendo Switch", "Switch"), ("Wii U", "Wii U"), ("3DS", "3DS"), ("PC", "PC"), ("Mac", "Mac")]


def _parts(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split("|")]


def category_text(text: str | None) -> str:
    parts = _parts(text)
    return parts[1] if len(parts) > 1 else ""


def categories(text: str | None, k: int = 3) -> list[str]:
    """Breadcrumbs under 'Video Games', e.g. ['Nintendo Switch', 'Games']."""
    return [c.strip() for c in category_text(text).split(">") if c.strip() and c.strip() != "Video Games"][:k]


def platforms(title: str | None, text: str | None, k: int = 2) -> list[str]:
    """Short platform names found in the title or the category path, at most k."""
    hay = f"{title or ''} {category_text(text)}"
    return [short for long_, short in PLATFORMS if long_ in hay][:k]


def description(text: str | None, max_chars: int) -> str:
    """The last ' | ' field of the gold text when the text has one (more than 3 fields), cut to max_chars."""
    parts = _parts(text)
    return (parts[-1] if len(parts) > 3 else "")[:max_chars]


def first_available(dfa_ts: int | None) -> date | None:
    """Amazon 'date first available' (seconds since epoch, <= 0 or None = unknown). NOT a release date."""
    return datetime.fromtimestamp(int(dfa_ts), tz=timezone.utc).date() if dfa_ts and dfa_ts > 0 else None
