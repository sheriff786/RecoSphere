"""feedback_service and audit_service."""
import uuid

import pytest

from app_api.models import FeedbackKind, User, UserKind, UserRole
from app_api.schemas.audit import AuditIn
from app_api.schemas.feedback import FeedbackIn
from app_api.services import audit_service as audit
from app_api.services import feedback_service as fb
from app_api.services.errors import NotFound


def new_user(db, role=UserRole.player) -> uuid.UUID:
    u = User(kind=UserKind.registered, email=f"{uuid.uuid4().hex[:8]}@example.test", password_hash="x",
             display_name="T", role=role)
    db.add(u)
    db.commit()        # like a real app: the user exists from an earlier request
    return u.id


def test_one_current_feedback_value_and_hidden_ids(db):
    uid = new_user(db)
    fb.set_feedback(db, uid, "G-ZELDA", FeedbackIn(kind=FeedbackKind.like))
    assert fb.hidden_item_ids(db, uid) == []
    out = fb.set_feedback(db, uid, "G-ZELDA", FeedbackIn(kind=FeedbackKind.not_interested))   # changed mind
    assert out.kind == FeedbackKind.not_interested
    fb.set_feedback(db, uid, "G-MARIO", FeedbackIn(kind=FeedbackKind.dislike))
    assert fb.hidden_item_ids(db, uid) == ["G-MARIO", "G-ZELDA"]
    assert fb.clear(db, uid, "G-ZELDA") and fb.hidden_item_ids(db, uid) == ["G-MARIO"]
    assert fb.clear(db, uid, "G-ZELDA") is False


def test_feedback_on_unknown_item(db):
    with pytest.raises(NotFound):
        fb.set_feedback(db, new_user(db), "NO-SUCH", FeedbackIn(kind=FeedbackKind.like))


def test_audit_record_and_recent(db):
    admin = new_user(db, UserRole.admin)
    a = audit.record(db, admin, AuditIn(action="model.rollback", target="20261111-171751", details={"from": "x"}))
    b = audit.record(db, admin, AuditIn(action="experiment.stop", target="new-slots-2-vs-0"))
    assert a.actor_id == admin and a.details == {"from": "x"} and b.details == {}
    ids = [r.id for r in audit.recent(db, k=10)]
    assert ids.index(b.id) < ids.index(a.id)                # newest first
