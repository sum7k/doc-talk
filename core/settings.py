from functools import lru_cache
from typing import Annotated, Literal, Union

from fastapi import Depends
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JWTConfig(BaseSettings):
    """JWT configuration with validation."""

    secret_key: str = Field(
        default="",
        min_length=0,
        description="JWT secret key (min 32 chars for production)",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    model_config = SettingsConfigDict(env_prefix="JWT_")


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = ""
    user: str = ""
    password: str = ""
    url_override: str = (
        ""  # Optional: Override constructed URL (for testing with SQLite)
    )
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    model_config = SettingsConfigDict(env_prefix="DB_")

    @computed_field
    def url(self) -> str:
        """Construct database URL from individual components.

        If url_override is set (e.g., for testing with SQLite), use that instead.
        """
        if self.url_override:
            return self.url_override
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @field_validator("url_override")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v:
            return v
        if not (
            v.startswith("postgresql://")
            or v.startswith("postgresql+asyncpg://")
            or v.startswith("sqlite")
        ):
            raise ValueError(
                "Database URL must be a valid PostgreSQL or SQLite connection string"
            )
        return v


Provider = Literal["openai", "local"]


class EmbeddingsConfig(BaseSettings):
    provider: Provider = "openai"
    model: str = "text-embedding-3-small"
    version: str = "v1"
    timeout: float = 30.0
    batch_size: int = 100
    namespace: str = "doc-talk"

    # provider-specific (used only when relevant)
    api_key: str | None = None
    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")


class VectorStoreConfigBase(BaseSettings):
    """Vector store configuration."""


class PgVectorConfig(VectorStoreConfigBase):
    backend: str = "pgvector"
    dsn: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20
    model_config = SettingsConfigDict(env_prefix="PG_VECTOR_")


class QdrantConfig(VectorStoreConfigBase):
    backend: str = "qdrant"
    url: str
    api_key: str | None = None
    collection_name: str = "embeddings"
    vector_size: int
    distance: Literal["cosine", "euclidean", "dot"] = "cosine"
    on_disk: bool = False
    model_config = SettingsConfigDict(env_prefix="QDRANT_")


VectorStoreConfig = Union[PgVectorConfig, QdrantConfig]


class Settings(BaseSettings):
    """Application settings."""

    service_name: str = "doc-talk"
    otlp_endpoint: str = ""  # Jaeger OTLP endpoint
    log_level: str = "INFO"

    # Nested configurations
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    vector_store: VectorStoreConfig = Field(default_factory=PgVectorConfig)
    embeddings_config: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]
