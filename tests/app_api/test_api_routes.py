"""app_api over HTTP (P1 phase D): the real FastAPI app, both DB dependencies bound to the test's rolled-back
session, the real service-token check with a test token."""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app_api import deps
from app_api.cli import create_admin
from app_api.db.session import get_autocommit_db, get_db
from app_api.main import app
from app_api.services import auth_service

TOKEN = "test-service-token"


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(auth_service, "BCRYPT_ROUNDS", 4)
    monkeypatch.setattr(deps, "get_settings", lambda: SimpleNamespace(service_token=TOKEN))
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_autocommit_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def h(sid=None, token=TOKEN) -> dict:
    out = {"X-Service-Token": token} if token else {}
    if sid:
        out["X-Session-Id"] = str(sid)
    return out


def signup(client, email="player@example.test", pw="correct-horse-1") -> dict:
    r = client.post("/auth/signup", json={"email": email, "password": pw}, headers=h())
    assert r.status_code == 201, r.text
    return r.json()


def test_service_token_is_required_everywhere_but_health(client):
    assert client.get("/health").status_code == 200                       # Docker / monitoring, no token
    for headers in ({}, h(token="wrong-token")):
        assert client.get("/auth/me", headers=headers).status_code == 401
        assert client.post("/auth/signup", json={"email": "a@b.co", "password": "x" * 9}, headers=headers).status_code == 401


def test_service_token_unset_fails_closed(client, monkeypatch):
    monkeypatch.setattr(deps, "get_settings", lambda: SimpleNamespace(service_token=None))
    assert client.get("/auth/me", headers=h()).status_code == 401


def test_signup_me_logout(client):
    s = signup(client)
    assert s["needs_onboarding"] and s["user"]["kind"] == "registered"
    me = client.get("/auth/me", headers=h(s["session_id"]))
    assert me.status_code == 200 and me.json()["user"]["id"] == s["user"]["id"]
    assert client.post("/auth/logout", headers=h(s["session_id"])).json() == {"revoked": True}
    assert client.get("/auth/me", headers=h(s["session_id"])).status_code == 401    # the old cookie is dead
    assert client.post("/auth/logout", headers=h(s["session_id"])).json() == {"revoked": False}
    assert client.get("/auth/me", headers=h()).status_code == 401                    # no session id
    assert client.get("/auth/me", headers=h(uuid.uuid4())).status_code == 401


def test_signup_and_login_errors(client):
    signup(client)
    dup = client.post("/auth/signup", json={"email": "PLAYER@example.test", "password": "another-pw-1"}, headers=h())
    assert dup.status_code == 409 and dup.json()["code"] == "conflict"
    assert client.post("/auth/signup", json={"email": "not-an-email", "password": "x" * 9}, headers=h()).status_code == 422
    assert client.post("/auth/login", json={"email": "player@example.test", "password": "nope-nope"},
                       headers=h()).status_code == 401
    ok = client.post("/auth/login", json={"email": "player@example.test", "password": "correct-horse-1"}, headers=h())
    assert ok.status_code == 200 and ok.json()["needs_onboarding"] is True        # never picked games yet


def test_demo_login(client):
    r = client.post("/auth/demo", json={"dataset_user_id": "AEPQB547GDARJWCYCDJO4X2WH5CQ"}, headers=h())
    assert r.status_code == 200 and r.json()["user"]["kind"] == "demo" and not r.json()["needs_onboarding"]


def test_onboarding_play_and_library(client):
    sid = signup(client)["session_id"]
    bad = client.post("/library/picks", json={"item_ids": ["G-ZELDA", "G-MARIO", "A-PAD"]}, headers=h(sid))
    assert bad.status_code == 422 and bad.json()["code"] == "invalid"             # an accessory is not a game
    ok = client.post("/library/picks", json={"item_ids": ["G-ZELDA", "G-MARIO", "G-POKEMON"]}, headers=h(sid))
    assert sorted(ok.json()["added"]) == ["G-MARIO", "G-POKEMON", "G-ZELDA"]
    assert client.get("/auth/me", headers=h(sid)).json()["needs_onboarding"] is False

    cid = str(uuid.uuid4())
    start = client.post("/plays/start", json={"item_id": "G-DLC", "client_event_id": cid, "surface": "home:top_picks"},
                        headers=h(sid))
    assert start.status_code == 200 and start.json()["first_time"] is True
    again = client.post("/plays/start", json={"item_id": "G-DLC", "client_event_id": cid}, headers=h(sid))
    assert again.json()["play_id"] == start.json()["play_id"]                     # a retried request is a no-op
    end = client.post("/plays/end", json={"client_event_id": cid, "seconds": 42}, headers=h(sid))
    assert end.json()["seconds_played"] == 42
    assert client.post("/plays/start", json={"item_id": "NO-SUCH", "client_event_id": str(uuid.uuid4())},
                       headers=h(sid)).status_code == 404

    lib = client.get("/library", params={"k": 5}, headers=h(sid)).json()
    assert lib[0]["item_id"] == "G-DLC" and lib[0]["seconds_played"] == 42
    assert client.get("/library/stats", headers=h(sid)).json() == {"games": 4, "plays": 1, "seconds_played": 42}

    other = signup(client, email="other@example.test")["session_id"]
    assert client.post("/plays/end", json={"client_event_id": cid, "seconds": 1}, headers=h(other)).status_code == 409


def test_feedback(client):
    sid = signup(client)["session_id"]
    assert client.put("/feedback/G-MARIO", json={"kind": "not_interested"}, headers=h(sid)).status_code == 200
    assert client.get("/feedback/hidden", headers=h(sid)).json() == {"item_ids": ["G-MARIO"]}
    assert client.put("/feedback/G-MARIO", json={"kind": "meh"}, headers=h(sid)).status_code == 422
    assert client.put("/feedback/NO-SUCH", json={"kind": "like"}, headers=h(sid)).status_code == 404
    assert client.delete("/feedback/G-MARIO", headers=h(sid)).json() == {"cleared": True}


def test_admin_audit_is_admins_only(client, db, monkeypatch):
    player = signup(client)["session_id"]
    assert client.get("/admin/audit", headers=h(player)).status_code == 403
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-password-1")
    assert create_admin(db, "boss@example.test", "admin-password-1") == "created"
    assert create_admin(db, "boss@example.test", None) == "promoted"               # idempotent
    admin = client.post("/auth/login", json={"email": "boss@example.test", "password": "admin-password-1"},
                        headers=h()).json()["session_id"]
    rec = client.post("/admin/audit", json={"action": "model.rollback", "target": "run-1"}, headers=h(admin))
    assert rec.status_code == 201
    assert client.get("/admin/audit", params={"k": 5}, headers=h(admin)).json()[0]["id"] == rec.json()["id"]
