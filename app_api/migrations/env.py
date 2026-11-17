"""Alembic environment for app_api. Models (Base.metadata) are the source of truth from P1 phase A on.

Connection: a caller may pass an open one in config.attributes["connection"] (tests do, inside a
rolled-back transaction); otherwise DATABASE_URL_DIRECT from the environment / app_api/.env.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy.pool import NullPool

import app_api.models  # noqa: F401  (registers every table on Base.metadata)
from app_api.config import get_settings
from app_api.db.base import Base, include_name
from app_api.db.session import make_engine

config = context.config
if config.config_file_name is not None and not config.attributes.get("connection"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

OPTS = dict(target_metadata=Base.metadata, include_name=include_name, compare_type=True,
            compare_server_default=True, transaction_per_migration=True)


def run_offline() -> None:
    context.configure(url=get_settings().migrations_url, literal_binds=True, **OPTS)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    given = config.attributes.get("connection")
    if given is not None:
        # a caller-owned connection may point Alembic at its own version table (tests use a scratch schema;
        # without this Alembic would find another schema's alembic_version on the search_path and skip work)
        context.configure(connection=given, version_table_schema=config.attributes.get("version_table_schema"), **OPTS)
        context.run_migrations()           # the caller owns the transaction
        return
    engine = make_engine(get_settings().migrations_url, poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, **OPTS)
        with context.begin_transaction():
            context.run_migrations()


run_offline() if context.is_offline_mode() else run_online()
