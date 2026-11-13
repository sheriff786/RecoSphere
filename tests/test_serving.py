"""Fast unit tests for serving pieces (no models needed).  Run: python -m pytest tests/test_serving.py -q"""
import numpy as np
import polars as pl

from ranking.pipeline import new_item_slots, tie_safe_top
from serve.feature_store import InMemoryStore, bootstrap_records


def test_tie_safe_top_is_deterministic():
    items = np.array([5, 3, 9, 1]); scores = np.array([0.5, 0.9, 0.5, 0.1], np.float32)
    assert tie_safe_top(items, scores, 3).tolist() == [3, 5, 9]          # tie 0.5 -> smaller id first
    assert tie_safe_top(items[::-1], scores[::-1], 3).tolist() == [3, 5, 9]


def test_new_item_slots():
    is_new = np.zeros(20, bool); is_new[[15, 17]] = True
    top = np.arange(20)                                                  # 0..9 old, new ones at 15 and 17
    out = new_item_slots(top, is_new, k=10, min_new=2)
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 15, 17]              # lowest old items replaced, order kept
    assert new_item_slots(top, np.ones(20, bool), k=10, min_new=2).tolist() == list(range(10))


def test_bootstrap_and_events_follow_batch_rules():
    hist = pl.DataFrame({"uid": [0, 0, 0, 1], "iid": [2, 1, 3, 4], "ts": [30, 10, 20, 5], "rating": [5.0, 1.0, 3.0, 4.0]})
    umap = pl.DataFrame({"uid": [0, 1, 2], "user_id": ["a", "b", "c"]})
    store_of = np.array([0, 1, 1, 2, 3])
    s = InMemoryStore()
    s.load(bootstrap_records(hist, umap, store_of, hist_len=2), n_users=3, version="v1")
    s.register_users(umap)
    _, r = s.get("a")
    assert r["hist"] == [3, 2] and r["n_hist"] == 3 and r["first"] == 10 and r["last"] == 30   # last 2 by time
    assert r["stores"] == {1: 2, 2: 1} and abs(r["sum_rating"] - 9.0) < 1e-9
    assert s.get("c") == (2, None)                                        # known id, no history -> fallback
    assert s.add_event("a", 1, 40, 5.0, 1, 2)[1] is False                 # already seen -> dedup
    uid, added = s.add_event("new", 4, 50, 5.0, 3, 2)
    assert added and uid == 3 and s.get("new")[1]["hist"] == [4]
