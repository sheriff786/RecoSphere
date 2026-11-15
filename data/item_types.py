"""Item type for every catalog item: game | accessory | other (P0.5).

Amazon's Video_Games category mixes games with controllers, cases, chargers, figures, guides and even books,
and users buy all of them. The model learned "next purchase", so it recommends all of them; the product (a
game-streaming app) must only offer games in game surfaces. This is a SERVING rule: the model is not
retrained and item_type is not a ranker feature (training/serving parity is unchanged).

Rules, first match wins (path = category path, e.g. Video Games > Nintendo Switch > Games):
  1. main_category outside gaming (Books, Movies & TV, ...)                      -> other
  2. title says it is not a video game (strategy guide, membership, gift card)   -> other
  3. path has a hardware node (Accessories, Consoles, ...), or a console bundle  -> accessory
  4. title has a STRONG accessory word (controller, charger, amiibo, figure ...) -> accessory
     (beats a "Games" path: Amazon files screen protectors and screwdrivers under Games)
  5. toy-figure franchise (Skylanders, Disney Infinity, LEGO Dimensions), unless the title is a game
     listing ("... - Nintendo 3DS", "Starter Pack")                              -> accessory
  6. path has a games node (Games, Downloadable Content, ...)                    -> game
  7. no Games path and a tabletop word (board game, miniatures), not a listing   -> other
  8. no path and a WEAK accessory word (case, stand, toy, remote ...), unless the title is a game listing
                                                                                 -> accessory
  9. nothing matched                                                             -> game
The category path is the 2nd " | " field of the gold item text ("title | Video Games > PC > Games | ...").

    python -m data.item_types [--write]   # counts for the catalog; --write saves gold/<cat>/item_types.parquet
"""
from __future__ import annotations

import re

GAME_NODES = {"Games", "Digital Video Games", "PC Games", "Linux Games", "Xbox 360 Kinect Games", "The Sims Games",
              "Downloadable Content", "PlayStation Digital Content", "Game Genre of the Month",
              "Value Games Under $20", "Digital Games", "Nintendo eShop", "PlayStation Network", "Xbox Live"}
HW_NODES = {"Accessories", "Consoles", "Microconsoles", "Standalone Hardware", "PlayStation VR Hardware",
            "PlayStation Move Hardware & Accessories", "Interactive Gaming Figures", "Subscription Cards"}
NON_GAMING_MAIN = {"Books", "Buy a Kindle", "Movies & TV", "Digital Music", "Kindle Store", "Audible Audiobooks",
                   "CDs & Vinyl", "Grocery", "Beauty & Personal Care", "Health & Personal Care",
                   "Arts, Crafts & Sewing", "Musical Instruments", "AMAZON FASHION", "Handmade"}


def _words(*ws: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(ws) + r")\b", re.I)


NOT_A_VIDEO_GAME = _words(
    r"strategy guides?", r"game guides?", r"official guide", r"guide ?book", r"game pass", r"memberships?",
    r"subscriptions?", r"gift cards?", r"points cards?", r"prepaid cards?", r"wallet top-?up", r"novel",
    r"quicken", r"turbotax")
TABLETOP = _words(r"board ?games?", r"card games?", r"tabletop", r"miniatures", r"jigsaw")
CONSOLE_BUNDLE = re.compile(
    r"^(sony\s+)?(playstation ?\d( pro| slim)?|ps\d( pro| slim)?|xbox (one|360|series [xs])( [sx])?|nintendo switch"
    r"( lite| oled)?|wii u?|nintendo (3ds|2ds|ds)( xl)?)\b[^-–|]*\b(bundle|console|system)\b", re.I)
STRONG_ACCESSORY = _words(
    r"controllers?", r"gamepads?", r"joysticks?", r"arcade sticks?", r"fight ?sticks?", r"racing wheels?",
    r"steering wheels?", r"pedals?", r"flight controls?", r"yoke", r"headsets?", r"headphones?", r"earbuds?",
    r"earphones?", r"chargers?", r"charging", r"power station", r"cables?", r"cords?", r"adapters?", r"converters?",
    r"hdmi", r"power supply", r"ac adapter", r"carrying case", r"travel case", r"hard case", r"screen protectors?",
    r"protectors?", r"tempered glass", r"skins?", r"decals?", r"faceplates?", r"batter(?:y|ies)", r"thumb ?grips?",
    r"grips?", r"docks?", r"docking", r"keyboards?", r"mouse", r"mice", r"mouse ?pads?", r"memory cards?",
    r"sd cards?", r"micro ?sd", r"hard drives?", r"ssd", r"cooling", r"cooler", r"stylus(?:es)?", r"sensor bar",
    r"amiibo", r"figures?", r"figurines?", r"plush(?:ies)?", r"statues?", r"posters?", r"t-?shirts?", r"hoodies?",
    r"mugs?", r"keychains?", r"lanyards?", r"wrist ?straps?", r"screwdrivers?", r"tool ?kit", r"replacement",
    r"repair", r"cleaning kit", r"vr headset", r"consoles?", r"system bundle", r"silicone cover", r"analog caps?",
    r"charging stand", r"wall mount", r"head strap", r"accessor(?:y|ies)", r"video game system", r"game system",
    r"illuminator", r"golf club", r"handles?", r"oculus touch", r"playstand", r"cushion", r"\d+ ?(?:gb|tb)")
FIGURE_FRANCHISE = _words(r"skylanders", r"disney infinity", r"lego dimensions", r"power discs?", r"starlink")
WEAK_ACCESSORY = _words(
    r"cases?", r"sleeves?", r"pouch(?:es)?", r"bags?", r"backpacks?", r"stands?", r"mounts?", r"holders?",
    r"remotes?", r"mic", r"microphones?", r"speakers?", r"toys?", r"buttons?", r"triggers", r"shells?", r"housing",
    r"storage", r"organizers?", r"racks?", r"camera", r"lens", r"led", r"light bar", r"fans?", r"straps?",
    r"wristbands?", r"stickers?", r"joy-?cons?", r"crate", r"box set of")
GAME_LISTING = re.compile(
    r"(\s[-–]\s*(nintendo switch|switch|playstation ?\d|ps\d|ps vita|xbox( one| 360| series x(\s?\|\s?s)?)?|"
    r"pc|mac|wii u|wii|nintendo 3ds|3ds|nintendo ds|ds|gamecube|game ?boy( advance| color)?|psp|"
    r"nintendo 64|n64|sega genesis|dreamcast|super nintendo|snes|nes)\b[^-–]*$)|\[(digital code|online game code|"
    r"download)\]|\b(standard|deluxe|gold|ultimate|collector'?s|game of the year|goty|complete) edition\b|"
    r"\b(season|expansion) pass\b|\bstarter pack\b", re.I)

TYPES = ("game", "accessory", "other")


def category_path(text: str | None) -> list[str]:
    """Category path from the gold item text: "title | Video Games > PC > Games | ..." -> [..levels..]."""
    parts = (text or "").split(" | ")
    if len(parts) < 2 or not parts[1].strip().startswith("Video Games"):
        return []
    return [c.strip() for c in parts[1].split(">") if c.strip()]


def classify(title: str | None, path: list[str] | None, main_category: str | None = None) -> tuple[str, str]:
    """-> (item_type, reason). Rule order: see the module docstring."""
    title = title or ""
    nodes = set((path or [])[1:])
    listing = bool(GAME_LISTING.search(title))
    if main_category in NON_GAMING_MAIN:
        return "other", f"main_category:{main_category}"
    if m := NOT_A_VIDEO_GAME.search(title):
        return "other", f"title:{m.group(0).lower()}"
    if nodes & HW_NODES:
        return "accessory", "path:" + sorted(nodes & HW_NODES)[0]
    if CONSOLE_BUNDLE.search(title):
        return "accessory", "title:console bundle"
    if m := STRONG_ACCESSORY.search(title):
        return "accessory", f"title:{m.group(0).lower()}"
    if (m := FIGURE_FRANCHISE.search(title)) and not listing:
        return "accessory", f"figure:{m.group(0).lower()}"
    if nodes & GAME_NODES:
        return "game", "path:" + sorted(nodes & GAME_NODES)[0]
    if (m := TABLETOP.search(title)) and not listing:
        return "other", f"title:{m.group(0).lower()}"
    if (m := WEAK_ACCESSORY.search(title)) and not listing:
        return "accessory", f"weak:{m.group(0).lower()}"
    return "game", "default"


def classify_items(titles: list[str], texts: list[str], main_categories: list[str | None] | None = None) -> list[str]:
    """item_type for each item, from the gold title + text (+ silver main_category when available)."""
    mc = main_categories or [None] * len(titles)
    return [classify(t, category_path(x), m)[0] for t, x, m in zip(titles, texts, mc)]


def main_categories_for(item_ids: list[str], category: str, data_root: str = "data") -> list[str | None] | None:
    """Silver main_category per item id (None when the silver layer is not on disk, e.g. CI / synthetic)."""
    import glob

    import polars as pl

    files = glob.glob(f"{data_root}/silver/products/category={category}/*.parquet")
    if not files:
        return None
    s = pl.concat([pl.read_parquet(f, columns=["item_id", "main_category"]) for f in files]).unique("item_id")
    return pl.DataFrame({"item_id": item_ids}).join(s, on="item_id", how="left")["main_category"].to_list()


def load_item_types(gold_dir, item_ids: list[str], titles: list[str], texts: list[str]) -> tuple[list[str], str]:
    """item_type per item for the engine. Prefers <gold>/item_types.parquet (written by `--write`, uses the
    silver main_category); otherwise classifies from the gold title + text. -> (types, source)."""
    from pathlib import Path

    import polars as pl

    p = Path(gold_dir) / "item_types.parquet"
    if p.exists():
        t = dict(pl.read_parquet(p, columns=["item_id", "item_type"]).iter_rows())
        if all(x in t for x in item_ids):
            return [t[x] for x in item_ids], str(p)
    return classify_items(titles, texts), "classified at startup (no item_types.parquet)"


if __name__ == "__main__":
    import argparse

    import polars as pl

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("category", nargs="?", default="Video_Games")
    ap.add_argument("--data", default="data")
    ap.add_argument("--write", action="store_true", help="write <data>/gold/<category>/item_types.parquet")
    a = ap.parse_args()
    f = pl.read_parquet(f"{a.data}/gold/{a.category}/item_features.parquet").sort("iid")
    mc = main_categories_for(f["item_id"].to_list(), a.category, a.data)
    rows = [classify(t, category_path(x), m) for t, x, m in
            zip(f["title"].fill_null("").to_list(), f["text"].fill_null("").to_list(), mc or [None] * f.height)]
    out = f.select("item_id").with_columns(pl.Series("item_type", [r[0] for r in rows]),
                                           pl.Series("reason", [r[1] for r in rows]))
    print(f"{f.height:,} items, main_category from silver: {'yes' if mc else 'no'}")
    print(out.group_by("item_type").len().sort("len", descending=True))
    print(out.with_columns(pl.col("reason").str.split(":").list.first().alias("signal"))
          .group_by("item_type", "signal").len().sort(["item_type", "len"], descending=[False, True]))
    if a.write:
        dst = f"{a.data}/gold/{a.category}/item_types.parquet"
        out.write_parquet(dst)
        print("wrote", dst)
