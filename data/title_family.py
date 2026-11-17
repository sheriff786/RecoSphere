"""title_family: one key per game across its Amazon listings (P1 phase C, P0.5 item 4: edition dedup).

    "The Legend of Zelda: Breath of the Wild - Nintendo Switch"        ┐
    "The Legend of Zelda: Breath of the Wild - Wii U [Digital Code]"   ├─> "legend of zelda breath of the wild"
    "The Legend of Zelda: Breath of the Wild - Explorer's Edition"     ┘

What is removed: accents and case; anything in [brackets] or (parentheses) ("[Digital Code]", "(PS4)",
"(European Version)"); a trailing " - ..." part made only of platform / edition / store words, or a short
named edition (" - Explorer's Edition"); "<word> Edition"
for edition words (standard, deluxe, gold, goty ...); region / old "Version"; a leading "the"; punctuation.
What is KEPT on purpose: numbers ("FIFA 21" != "FIFA 22", "Super Mario Bros. 3"), subtitles, and words that name
a game on their own ("Pokemon Gold Version" != "Pokemon Silver Version": only "<word> Edition" drops "gold").
False merges (two different games, one family) are worse than missed merges, so the rules stay conservative.
"""
from __future__ import annotations

import re
import unicodedata

PLATFORM = (r"pc|mac|pc/mac|pc ?dvd|steam|windows|linux|ps ?[1-5]|psx|ps ?vita|vita|psp|sony psp|"
            r"playstation(?: ?[1-5]| vita| portable)?|xbox(?: one| 360| series x(?: ?\| ?s)?| series s)?|xb1|x360|"
            r"nintendo(?: switch| wii u| wii| ds| 3ds| 2ds| 64| nes| super nes| gamecube| game ?boy(?: advance| color)?)?|"
            r"switch|nsw|wii u|wii|3ds|2ds|ds|nds|gamecube|ngc|game ?boy(?: advance| color)?|gba|gbc|n64|nes|snes|"
            r"super nes|super nintendo|sega(?: genesis| dreamcast| saturn| game gear)?|genesis|dreamcast|saturn|"
            r"atari ?\d*|turbografx|neo ?geo")
EDITION_WORD = (r"standard|deluxe|digital deluxe|gold|ultimate|special|complete|limited|definitive|premium|launch|"
                r"day one|day 1|collector'?s|collectors|game of the year|goty|anniversary|platinum|legendary|"
                r"steelbook|enhanced|bonus|signature|super deluxe|digital")
NOISE = (rf"(?:{PLATFORM}|{EDITION_WORD}|edition|version|digital code|online game code|download|instant access|"
         r"code|game only|jewel case|import|japan(?:ese)?|uk|us|usa|eu|european|english|asian|region free|"
         r"old version|nintendo selects|greatest hits|platinum hits|player'?s choice|best|the best|"
         r"physical|standard|and|&|/|\+|,)")

_BRACKETS = re.compile(r"[\[(][^\])]*[\])]")
_DASH_TAIL = re.compile(r"\s[-–—]\s+([^-–—]*)$")
_EDITION = re.compile(rf"\b(?:{EDITION_WORD})\s+edition\b|\bedition\b$")
_VERSION = re.compile(r"\b(?:old|european|us|uk|japanese|english|asian|international|north american)\s+version\b")
_ONLY_NOISE = re.compile(rf"^(?:\s*(?:{NOISE})\s*)+$")
_NAMED_EDITION = re.compile(r"^(?:[\w'.]+\s+){1,3}edition(?:\s+(?:" + PLATFORM + r"))?$")  # "- explorer's edition"
_PUNCT = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    """Lowercase, accents off (Pokémon -> pokemon), dashes and quotes made ASCII FIRST (the ASCII step would
    silently delete an en dash and with it the ' - Xbox One' separator)."""
    text = text.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'").replace("`", "'")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


ERA_GAP_YEARS = 5


def families(titles: list[str | None], years: list[int | None], gap: int = ERA_GAP_YEARS) -> list[str]:
    """title_family per item, split by era: inside one title family, items sorted by first-available year
    start a new family where two consecutive years are more than `gap` apart ("doom" and "doom~2014").

    Why: identical titles can be different games (Doom 1993 vs the 2016 reboot, Resident Evil 2 vs its 2019
    remake). Real catalog 2026-11-17: 234 multi-listing families span more than 5 years. Items without a year
    stay in the first era. Deterministic from (title, year), which the engine also has, so engine rows and the
    games table agree. Cost: a listing with a wrong date can split off = a missed merge (a duplicate shows),
    never a false one.
    """
    base = [title_family(t) for t in titles]
    out = list(base)
    by_family: dict[str, list[int]] = {}
    for i, b in enumerate(base):
        by_family.setdefault(b, []).append(i)
    for b, idx in by_family.items():
        dated = sorted((years[i], i) for i in idx if years[i] is not None)
        era_start = None
        prev = None
        for y, i in dated:
            if prev is not None and y - prev > gap:
                era_start = y                                   # a new era begins here
            if era_start is not None:
                out[i] = f"{b}~{era_start}"
            prev = y
    return out


def title_family(title: str | None) -> str:
    t = _fold(title or "")
    t = _BRACKETS.sub(" ", t)
    while (m := _DASH_TAIL.search(t)) and (_ONLY_NOISE.match(tail := m.group(1).strip() or "x")
                                            or _NAMED_EDITION.match(tail)):
        t = t[: m.start()]                                   # drop "- Nintendo Switch", "- Explorer's Edition" ...
    t = _EDITION.sub(" ", t)
    t = _VERSION.sub(" ", t)
    t = t.replace("&", " and ")
    t = _PUNCT.sub(" ", t).strip()
    t = re.sub(r"^the\s+", "", t)
    return t or _PUNCT.sub(" ", _fold(title or "")).strip()   # never empty: fall back to the plain title
