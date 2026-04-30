from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(alias="APP_NAME")
    app_env: str = Field(alias="APP_ENV")
    log_level: str = Field(alias="LOG_LEVEL")
    database_url: str = Field(alias="DATABASE_URL")
    embedding_dim: int = Field(alias="EMBEDDING_DIM")
    litellm_completion_model: str = Field(alias="LITELLM_COMPLETION_MODEL")
    litellm_embedding_model: str = Field(alias="LITELLM_EMBEDDING_MODEL")
    litellm_rerank_model: str | None = Field(default=None, alias="LITELLM_RERANK_MODEL")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    phoenix_tracing_enabled: bool = Field(default=False, alias="PHOENIX_TRACING_ENABLED")
    phoenix_collector_endpoint: str = Field(
        default="http://phoenix.docker.home.arpa:6006",
        alias="PHOENIX_COLLECTOR_ENDPOINT",
    )
    phoenix_project_name: str = Field(default="camillo", alias="PHOENIX_PROJECT_NAME")
    decay_rate: float = Field(alias="DECAY_RATE")
    recall_top_k: int = Field(alias="RECALL_TOP_K")
    recall_vector_limit: int = Field(alias="RECALL_VECTOR_LIMIT")
    recall_full_text_search_limit: int = Field(alias="RECALL_FULL_TEXT_SEARCH_LIMIT")
    hebbian_edge_threshold: float = Field(alias="HEBBIAN_EDGE_THRESHOLD")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for process-wide configuration."""
    return Settings()


settings = get_settings()
