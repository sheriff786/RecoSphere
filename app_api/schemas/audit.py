import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

from app_api.schemas.common import Out

Action = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[a-z_]+\.[a-z_]+$", max_length=64)]


class AuditIn(BaseModel):
    action: Action                       # 'model.rollback', 'experiment.stop', 'user.role_change'
    target: Annotated[str, StringConstraints(max_length=200)] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditOut(Out):
    id: int
    actor_id: uuid.UUID | None
    action: str
    target: str | None
    details: dict[str, Any]
    created_at: datetime
