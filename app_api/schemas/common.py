from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

ItemId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


def _password_fits_bcrypt(v: str) -> str:
    # bcrypt only looks at the first 72 BYTES; bcrypt >= 5 raises instead of truncating. Multi-byte
    # characters make "72 characters" the wrong limit, so check bytes.
    if len(v.encode("utf-8")) > 72:
        raise ValueError("password is longer than 72 bytes")
    return v


Password = Annotated[str, StringConstraints(min_length=8, max_length=72), AfterValidator(_password_fits_bcrypt)]


class Out(BaseModel):
    """Base for responses: built straight from ORM objects (model_validate(user))."""
    model_config = ConfigDict(from_attributes=True)
