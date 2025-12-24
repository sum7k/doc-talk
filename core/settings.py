from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JWTConfig(BaseSettings):
    """JWT configuration with validation."""

    secret_key: str = Field(
        ..., min_length=32, description="JWT secret key (min 32 chars)"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError(
                "JWT secret key must be at least 32 characters long for security"
            )
        return v


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(..., description="Database connection URL")
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    @field_validator("url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v:
            raise ValueError("Database URL cannot be empty")
        if not (
            v.startswith("postgresql://")
            or v.startswith("postgresql+asyncpg://")
            or v.startswith("sqlite")
        ):
            raise ValueError(
                "Database URL must be a valid PostgreSQL or SQLite connection string"
            )
        return v


class Settings(BaseSettings):
    """Application settings."""

    service_name: str = "fastapi-skeleton"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    db_url_override: str = (
        ""  # Optional: Override constructed URL (for testing with SQLite)
    )
    jwt_secret_key: str = ""
    otlp_endpoint: str = ""  # Jaeger OTLP endpoint
    log_level: str = "INFO"

    # tell pydantic where/how to load configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def db_url(self) -> str:
        """Construct database URL from individual components.

        If db_url_override is set (e.g., for testing with SQLite), use that instead.
        """
        if self.db_url_override:
            return self.db_url_override
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def get_jwt_config(self) -> JWTConfig:
        """Get JWT configuration from settings."""
        return JWTConfig(secret_key=self.jwt_secret_key)

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration from settings."""
        return DatabaseConfig(url=self.db_url)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]
