"""Pydantic schemas reject bad input before any service runs (no database needed)."""
import uuid

import pytest
from pydantic import ValidationError

from app_api.schemas.audit import AuditIn
from app_api.schemas.auth import SignupIn
from app_api.schemas.library import PicksIn
from app_api.schemas.play import PlayEndIn, PlayStartIn


def test_password_limit_is_72_bytes_not_72_characters():
    SignupIn(email="a@b.co", password="x" * 72)
    with pytest.raises(ValidationError, match="72 bytes"):
        SignupIn(email="a@b.co", password="é" * 40)       # 40 characters, 80 bytes: bcrypt would refuse it
    with pytest.raises(ValidationError):
        SignupIn(email="a@b.co", password="short")


@pytest.mark.parametrize("email", ["no-at-sign", "a@b", "a b@c.de", ""])
def test_signup_rejects_bad_emails(email):
    with pytest.raises(ValidationError):
        SignupIn(email=email, password="long-enough-1")


def test_picks_need_3_to_20_unique_ids():
    PicksIn(item_ids=["a", "b", "c"])
    for bad in (["a", "b"], ["a", "a", "b"], [str(i) for i in range(21)]):
        with pytest.raises(ValidationError):
            PicksIn(item_ids=bad)


def test_play_inputs():
    PlayStartIn(item_id="G1", client_event_id=uuid.uuid4(), surface="home:top_picks")
    with pytest.raises(ValidationError):
        PlayStartIn(item_id="G1", client_event_id="not-a-uuid")
    with pytest.raises(ValidationError):
        PlayEndIn(client_event_id=uuid.uuid4(), seconds=-1)
    with pytest.raises(ValidationError):
        PlayEndIn(client_event_id=uuid.uuid4(), seconds=10**7)


def test_audit_action_is_namespaced():
    AuditIn(action="model.rollback", target="20261111-171751")
    with pytest.raises(ValidationError):
        AuditIn(action="drop everything")
