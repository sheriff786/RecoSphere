import uuid
from datetime import datetime

from app_api.models.enums import UserKind, UserRole
from app_api.schemas.common import Out


class UserOut(Out):
    id: uuid.UUID
    kind: UserKind
    email: str | None
    display_name: str
    role: UserRole
    dataset_user_id: str | None
    onboarded_at: datetime | None
    created_at: datetime

    @property
    def engine_user_id(self) -> str:
        """The id the recommendation engine knows this user by: the dataset id for demo players, app:<uuid> else."""
        return self.dataset_user_id if self.kind == UserKind.demo else f"app:{self.id}"
