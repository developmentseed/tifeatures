"""tifeatures config."""

from functools import lru_cache
from typing import Any, Dict, Optional

import pydantic
from starlite import CORSConfig


class _APISettings(pydantic.BaseSettings):
    """API settings"""

    name: str = "TiFeatures"
    cors_origins: str = "*"
    cachecontrol: str = "public, max-age=3600"

    @pydantic.validator("cors_origins")
    def parse_cors_origin(cls, v):
        """Parse CORS origins."""
        return [origin.strip() for origin in v.split(",")]

    class Config:
        """model config"""

        env_prefix = "TIFEATURES_"
        env_file = ".env"

    @property
    def cors_config(self) -> Optional[CORSConfig]:
        if self.cors_origins:
            return CORSConfig(allow_origins=self.cors_origins)

        return None


@lru_cache()
def APISettings() -> _APISettings:
    """This function returns a cached instance of the Settings object."""
    return _APISettings()


class _PostgresSettings(pydantic.BaseSettings):
    """Postgres-specific API settings.

    Attributes:
        postgres_user: postgres username.
        postgres_pass: postgres password.
        postgres_host: hostname for the connection.
        postgres_port: database port.
        postgres_dbname: database name.
    """

    postgres_user: Optional[str]
    postgres_pass: Optional[str]
    postgres_host: Optional[str]
    postgres_port: Optional[str]
    postgres_dbname: Optional[str]

    database_url: Optional[pydantic.PostgresDsn] = None

    db_min_conn_size: int = 1
    db_max_conn_size: int = 10
    db_max_queries: int = 50000
    db_max_inactive_conn_lifetime: float = 300

    class Config:
        """model config"""

        env_file = ".env"

    # https://github.com/tiangolo/full-stack-fastapi-postgresql/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/backend/app/app/core/config.py#L42
    @pydantic.validator("database_url", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v

        return pydantic.PostgresDsn.build(
            scheme="postgresql",
            user=values.get("postgres_user"),
            password=values.get("postgres_pass"),
            host=values.get("postgres_host", ""),
            port=values.get("postgres_port", 5432),
            path=f"/{values.get('postgres_dbname') or ''}",
        )


@lru_cache()
def PostgresSettings() -> _PostgresSettings:
    """Postgres Settings."""
    return _PostgresSettings()
