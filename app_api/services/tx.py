"""One unit of work = one transaction: commit at the end, roll back on ANY error, re-raise."""
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session


@contextmanager
def transaction(db: Session) -> Iterator[Session]:
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
