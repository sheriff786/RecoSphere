"""Settings from the environment (and the git-ignored app_api/.env for local runs)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / "app_api" / ".env", env_file_encoding="utf-8", extra="ignore")

    # Neon: the -pooler host (PgBouncer, transaction mode) for the app, the direct host for Alembic.
    database_url: str
    database_url_direct: str | None = None
    pool_size: int = 5
    max_overflow: int = 5

    @property
    def app_url(self) -> str:
        return sqlalchemy_url(self.database_url)

    @property
    def migrations_url(self) -> str:
        return sqlalchemy_url(self.database_url_direct or self.database_url)


def sqlalchemy_url(url: str) -> str:
    """postgresql://... -> postgresql+psycopg://... (SQLAlchemy needs the driver in the scheme)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
