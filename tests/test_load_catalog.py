"""scripts/load_catalog.build_rows: gold rows -> GameIn (no database needed)."""
import importlib.util
from datetime import date
from pathlib import Path

import polars as pl

spec = importlib.util.spec_from_file_location("load_catalog", Path(__file__).resolve().parents[1] / "scripts" / "load_catalog.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

FEATURES = pl.DataFrame({
    "iid": [0, 1, 2],
    "item_id": ["B1", "B2", "B3"],
    "title": ["Doom - PC", "Doom - PlayStation 4", "Pro Controller for Xbox One"],
    "text": ["Doom - PC | Video Games > PC > Games | f1 | f2 | " + "x" * 5000,
             "Doom - PlayStation 4 | Video Games > PlayStation 4 > Games | short",
             "Pro Controller | Video Games > Xbox One > Accessories | a | b | wireless pad"],
    "dfa_ts": [946684800, 1451606400, 0],             # 2000-01-01, 2016-01-01, unknown
})


def test_build_rows():
    rows = lc.build_rows(FEATURES, ["game", "game", "accessory"], {"B1": "id Software", "B3": None})
    doom_pc, doom_ps4, pad = rows
    assert doom_pc.title_family == "doom" and doom_ps4.title_family == "doom~2016"   # reboot: separate era
    assert doom_pc.platforms == ["PC"] and doom_ps4.platforms == ["PS4"] and pad.platforms == ["Xbox One"]
    assert doom_pc.listed_on == date(2000, 1, 1) and pad.listed_on is None
    assert len(doom_pc.description) == 2000                      # capped
    assert doom_ps4.description is None                         # gold text without a description field
    assert pad.item_type.value == "accessory" and pad.publisher is None and doom_pc.publisher == "id Software"
