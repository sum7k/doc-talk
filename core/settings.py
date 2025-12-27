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


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = ""
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    model_config = SettingsConfigDict(extra="ignore")

    @field_validator("url")
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


class PgVectorConfig(BaseSettings):
    backend: Literal["pgvector"] = "pgvector"
    dsn: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20


class QdrantConfig(BaseSettings):
    backend: Literal["qdrant"] = "qdrant"
    url: str | None = None  # Remote server URL
    path: str | None = None  # Local storage directory path
    api_key: str | None = None
    collection_name: str = "embeddings"
    vector_size: int = 1536
    distance: Literal["cosine", "euclidean", "dot"] = "cosine"
    on_disk: bool = False


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: str | None = None
    timeout: float = 60.0


class Settings(BaseSettings):
    """Application settings."""

    service_name: str = "doc-talk"
    otlp_endpoint: str = ""  # Jaeger OTLP endpoint
    log_level: str = "INFO"
    data_dir: str = ""

    # Nested configurations
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    pgvector: PgVectorConfig = Field(default_factory=PgVectorConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        env_parse_enums=True,
        # Note: .env variable expansion (${VAR}) requires python-dotenv
        # Install with: pip install python-dotenv
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


SettingsDep = Annotated[Settings, Depends(get_settings)]
