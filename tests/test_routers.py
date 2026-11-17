"""P0: engine routes after the serve/routers split, with a fake ModelManager (no models needed).
Covers the routes the smoke test does not hit (/catalog/items, /users/{id}/history, /admin/reload,
admin token checks). The lifespan is not run: STATE is filled by hand."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from serve.app import app
from serve.deps import STATE


class FakeProduct:
    def items(self, ids):
        return [{"item_id": i} for i in ids]

    def history(self, user_id, k, item_type=None):
        return [{"item_id": f"{user_id}-h{j}"} for j in range(k)]


class FakeMgr:
    def __init__(self, token=""):
        self.cfg = SimpleNamespace(admin_token=token)
        self.current = SimpleNamespace(_product=FakeProduct(), info=lambda: {"model_run": "r1"})
        self.experiments = SimpleNamespace(refresh=lambda force=False: {"status": "unchanged"})
        self.reload_calls = []

    def reload(self, reason, force=False):
        self.reload_calls.append((reason, force))
        return {"status": "unchanged", "model_run": "r1"}

    def info(self):
        return {"reloads": 0}


@pytest.fixture
def client():
    def make(token=""):
        STATE["mgr"] = FakeMgr(token)
        return TestClient(app), STATE["mgr"]
    yield make
    STATE.clear()


def test_catalog_items_splits_ids(client):
    c, _ = client()
    r = c.get("/catalog/items", params={"ids": "a,,b"})
    assert r.status_code == 200 and r.json() == {"items": [{"item_id": "a"}, {"item_id": "b"}]}


def test_user_history_k(client):
    c, _ = client()
    r = c.get("/users/u1/history", params={"k": 2})
    assert r.json() == {"user_id": "u1", "items": [{"item_id": "u1-h0"}, {"item_id": "u1-h1"}]}
    assert c.get("/users/u1/history", params={"k": 0}).status_code == 422     # Query(ge=1) still enforced


def test_health_merges_model_and_manager_info(client):
    c, _ = client()
    assert c.get("/health").json() == {"model_run": "r1", "reloads": 0}


def test_admin_reload_without_token_configured(client):
    c, mgr = client()
    r = c.post("/admin/reload", params={"force": "true"})
    assert r.status_code == 200 and r.json()["experiment"] == {"status": "unchanged"}
    assert mgr.reload_calls == [("admin request", True)]


@pytest.mark.parametrize("method,path", [("post", "/admin/reload"), ("get", "/admin/summary"),
                                         ("post", "/admin/rollback")])
def test_admin_routes_need_the_token(client, method, path):
    c, mgr = client(token="secret")
    assert getattr(c, method)(path).status_code == 403
    assert getattr(c, method)(path, headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert mgr.reload_calls == []                                            # nothing happened


def test_admin_reload_with_token(client):
    c, mgr = client(token="secret")
    assert c.post("/admin/reload", headers={"X-Admin-Token": "secret"}).status_code == 200
    assert mgr.reload_calls == [("admin request", False)]


def test_replay_history_route(client):
    """P1 phase F: POST /users/{id}/history replays oldest-first through add_event; GET on the same path still works."""
    from tests.test_serving import _fake_engine
    from serve.recommender import Recommender
    c, mgr = client()
    eng = _fake_engine()
    mgr.current.add_events = Recommender.add_events.__get__(eng)
    body = {"items": [{"item_id": "I1"}, {"item_id": "I2"}, {"item_id": "nope"}]}
    assert c.post("/users/app:x/history", json=body).json() == \
        {"user_id": "app:x", "recorded": 2, "already_seen": 0, "unknown": ["nope"]}
    assert c.post("/users/app:x/history", json=body).json()["recorded"] == 0
    assert eng.store.get("app:x")[1]["hist"] == [1, 2]
    assert c.post("/users/app:x/history", json={"items": [{"item_id": "I1"}] * 501}).status_code == 422
    assert c.get("/users/app:x/history", params={"k": 1}).status_code == 200
