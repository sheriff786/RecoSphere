import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from app_api.schemas.common import Out, Password
from app_api.schemas.user import UserOut

Email = Annotated[str, StringConstraints(strip_whitespace=True, max_length=254,
                                         pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
DisplayName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]


class SignupIn(BaseModel):
    email: Email
    password: Password
    display_name: DisplayName | None = None       # default: the part of the email before '@'


class LoginIn(BaseModel):
    email: Email
    password: Annotated[str, StringConstraints(min_length=1, max_length=200)]   # no length rule on login


class DemoLoginIn(BaseModel):
    dataset_user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class SessionOut(Out):
    session_id: uuid.UUID                          # goes into the cookie's JWT as `sid` (phase D)
    expires_at: datetime
    user: UserOut
    needs_onboarding: bool
