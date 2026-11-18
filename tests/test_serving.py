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


def _fake_engine():
    """Just enough of a Recommender for add_event/add_events (real InMemoryStore, no model)."""
    from types import SimpleNamespace
    s = InMemoryStore()
    s.load([], n_users=0, version="v1")
    s.register_users(pl.DataFrame({"uid": [], "user_id": []}, schema={"uid": pl.Int64, "user_id": pl.Utf8}))
    from serve.recommender import Recommender
    e = SimpleNamespace(store=s, t_ref=1000, hist_len=3, iid_of={f"I{i}": i for i in range(6)},
                        ctx=SimpleNamespace(store=np.array([0, 1, 1, 2, 2, 3])))
    e.add_event = Recommender.add_event.__get__(e)
    return e


def test_replay_equals_one_by_one_events():
    """P1 phase F: POST /users/{id}/history must leave the user exactly as the same /event calls would."""
    from serve.recommender import Recommender
    items = ["I1", "I2", "I1", "I3", "I4", "I5"]                     # a repeat, and more than hist_len
    a, b = _fake_engine(), _fake_engine()
    for it in items:
        Recommender.add_event(a, "app:u", it)
    r = Recommender.add_events(b, "app:u", [(it, None, None) for it in items])
    ra, rb = a.store.get("app:u")[1], b.store.get("app:u")[1]
    assert ra["seen"].tolist() == rb["seen"].tolist() == [1, 2, 3, 4, 5]
    assert {k: v for k, v in ra.items() if k != "seen"} == {k: v for k, v in rb.items() if k != "seen"}
    assert b.store.get("app:u")[1]["hist"] == [3, 4, 5]              # oldest -> newest, trimmed to hist_len
    assert r == {"user_id": "app:u", "recorded": 5, "already_seen": 1, "unknown": []}
    again = Recommender.add_events(b, "app:u", [("I1", None, None), ("GONE", None, None)])
    assert again["recorded"] == 0 and again["already_seen"] == 1 and again["unknown"] == ["GONE"]
    assert b.store.get("app:u")[1]["last"] == 1000                   # no ts -> the engine's frozen clock t_ref


def _fake_ranker(new_item_slots=0):
    """A Recommender minus the model: fixed candidate scores, so recommend()'s serving steps run for real."""
    from types import SimpleNamespace
    from serve.recommender import Recommender
    n = 12
    items, scores = np.arange(n, dtype=np.int64), np.linspace(1.0, 0.0, n).astype(np.float32)  # 0 best ... 11 worst
    e = SimpleNamespace(
        item_type=np.array(["game"] * 10 + ["accessory"] * 2), item_ids=[f"I{i}" for i in range(n)],
        titles=[f"T{i}" for i in range(n)], iid_of={f"I{i}": i for i in range(n)},
        fallback=np.arange(n - 1, -1, -1, dtype=np.int64),
        items=SimpleNamespace(is_new=np.array([False] * 8 + [True] * 4)),
        cfg=SimpleNamespace(new_item_slots=new_item_slots),
        store=SimpleNamespace(get=lambda u: (0, {"n_hist": 1}) if u == "known" else (None, None)),
        score_candidates=lambda rec, timings: (items, scores))
    for m in ("keep", "iids", "recommend"):
        setattr(e, m, getattr(Recommender, m).__get__(e))
    return e


def test_exclude_only_removes_the_hidden_games():
    """P1 phase G "Not interested": excluded games never come back; everything else keeps its order, and the
    list is refilled from below (a filter after ranking, like item_type)."""
    ids = lambda r: [x["item_id"] for x in r["items"]]
    e = _fake_ranker()
    assert ids(e.recommend("known", k=5, item_type="game")) == ["I0", "I1", "I2", "I3", "I4"]
    assert ids(e.recommend("known", k=5, item_type="game", exclude=["I0", "I3", "UNKNOWN"])) == ["I1", "I2", "I4", "I5", "I6"]
    # with new-item slots (I8-I11 are new; I10, I11 are accessories): hiding a new game frees its slot for the next
    s = _fake_ranker(new_item_slots=2)
    assert ids(s.recommend("known", k=5, item_type="game")) == ["I0", "I1", "I2", "I8", "I9"]
    assert ids(s.recommend("known", k=5, item_type="game", exclude=["I8"])) == ["I0", "I1", "I2", "I3", "I9"]
    # the popular fallback (no history) honours it too
    assert ids(e.recommend("nobody", k=3, exclude=["I11"])) == ["I10", "I9", "I8"]
