"""Declarative base for every model.

The naming convention reproduces Postgres' own default names (users_pkey, users_email_key,
library_user_id_fkey), so constraints created from the models are named exactly like the ones that
001_init.sql created. CHECK constraints always get an explicit name in the models.
"""
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "pk": "%(table_name)s_pkey",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "ix": "%(table_name)s_%(column_0_label)s_idx",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# Tables in the database that are not models: Alembic's own bookkeeping, and the legacy Node runner's table
# (web/db/migrate.mjs, removed in P1 phase H). Autogenerate must never propose to drop them.
NOT_MODELS = {"alembic_version", "schema_migrations"}


def include_name(name, type_, parent_names) -> bool:
    """Alembic `include_name` hook, shared by migrations/env.py and the drift test."""
    return not (type_ == "table" and name in NOT_MODELS)
