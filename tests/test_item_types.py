"""P0.5 item 1: game vs accessory classification (data/item_types.py) and the serving filter."""
import json
import os
import urllib.request
import uuid

import pytest

from data.item_types import category_path, classify, classify_items

G = ["Video Games", "Nintendo Switch", "Games"]
ACC = ["Video Games", "Nintendo Switch", "Accessories", "Controllers"]


@pytest.mark.parametrize("title,path,main,expected", [
    ("The Legend of Zelda: Breath of the Wild - Explorer's Edition - Nintendo Switch", G, "Video Games", "game"),
    ("The Legend of Zelda: Breath of the Wild - Wii U [Digital Code]", [], None, "game"),          # no path: default
    ("Chrono Trigger", [], None, "game"),                                                          # 'trigger' is not 'triggers'
    ("PowerA Enhanced Wired Controller for Xbox Series X|S", ACC, "Video Games", "accessory"),      # hardware path
    ("Youletao Y Screwdriver for Nintendo Switch", G, "Tools & Home Improvement", "accessory"),     # strong word beats Games path
    ("Nintendo amiibo - Corrin (Smash Brothers series)", [], None, "accessory"),
    ("Skylanders Giants: Chop Chop Series 2", [], None, "accessory"),                              # figure franchise
    ("Skylanders Giants - PlayStation 3", [], None, "game"),                                       # ... but a game listing
    ("PlayStation 4 + Bloodborne Bundle", [], None, "accessory"),                                  # console bundle
    ("PlayStation 3 Slim 120GB (Old Model)", [], None, "accessory"),                               # storage size = hardware
    ("$25 PlayStation Store Gift Card [Digital Code]", G, "Video Games", "other"),
    ("Two Worlds II Official Strategy Guide", [], None, "other"),
    ("Wonder", G, "Books", "other"),                                                               # Amazon files books under Games
    ("Star Wars Armada Invisible Hand EXPANSION PACK | Miniatures Battle Game", [], None, "other"),
    ("Pikachu Hard Pouch", [], None, "accessory"),                                                 # weak word, no path
])
def test_classify(title, path, main, expected):
    assert classify(title, path, main)[0] == expected, classify(title, path, main)


def test_category_path_from_gold_text():
    assert category_path("Wonder | Video Games > PC > Games | #1 bestseller") == ["Video Games", "PC", "Games"]
    assert category_path("Title only") == []
    assert category_path("Quest Dragon edition 3 | sword mode multiplayer") == []    # synthetic text: no path


def test_classify_items_is_aligned():
    out = classify_items(["Zelda - Nintendo Switch", "Pro Controller"], ["Zelda | Video Games > Nintendo Switch > Games",
                                                                          "Pro Controller | Video Games"])
    assert out == ["game", "accessory"]


# ---- acceptance test on the REAL engine (skipped unless one runs on E2E_ENGINE_URL with the real catalog) ----
ENGINE = os.environ.get("E2E_ENGINE_URL", "http://127.0.0.1:8000")
SEEDS = {"zelda breath of the wild": "Breath of the Wild", "pokemon sword": "Sword",
         "super mario 3d all-stars": "3D All-Stars"}


def _get(path):
    return json.load(urllib.request.urlopen(ENGINE + path, timeout=30))


def _real_engine():
    try:
        h = _get("/health")
    except Exception:
        return None
    return h if h.get("items", 0) > 20_000 and "item_types" in h else None


@pytest.mark.skipif(_real_engine() is None, reason="needs the engine with the real catalog on E2E_ENGINE_URL")
def test_real_onboarding_three_nintendo_games_gives_no_accessories():
    from urllib.parse import quote
    picks = []
    for q, must in SEEDS.items():
        hits = [x for x in _get(f"/catalog/search?q={quote(q)}&k=10&item_type=game")["items"] if must in x["title"]]
        assert hits, f"no game found for {q!r}"
        picks.append(hits[0]["item_id"])
    user = f"test:onboard-{uuid.uuid4().hex[:8]}"
    for item in picks:
        req = urllib.request.Request(ENGINE + "/event", data=json.dumps({"user_id": user, "item_id": item}).encode(),
                                     headers={"content-type": "application/json"})
        assert json.load(urllib.request.urlopen(req, timeout=30))["recorded"]
    games = _get(f"/recommend/{user}?k=20&item_type=game")
    assert games["source"] == "personalized"
    assert len(games["items"]) == 20
    assert [x for x in games["items"] if x["item_type"] != "game"] == []
