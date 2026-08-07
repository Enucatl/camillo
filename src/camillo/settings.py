from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_secret_file(path: str) -> str:
    """Read a Docker secret without letting surrounding whitespace leak into config.

    Args:
        path: Absolute or relative path to the mounted secret file.

    Returns:
        The stripped secret value.
    """
    return Path(path).read_text(encoding="utf-8").strip()


class Settings(BaseSettings):
    """Parse the small operational configuration surface for one corpus."""

    model_config = SettingsConfigDict(
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(alias="APP_NAME")
    app_env: str = Field(alias="APP_ENV")
    log_level: str = Field(alias="LOG_LEVEL")
    database_url: str = Field(default="", alias="DATABASE_URL")
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    postgres_password_file: str | None = Field(default=None, alias="POSTGRES_PASSWORD_FILE")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    embedding_dim: int = Field(alias="EMBEDDING_DIM")
    chat_model: str = Field(validation_alias="INFERENCE_CHAT_MODEL")
    embedding_model: str = Field(validation_alias="INFERENCE_EMBEDDING_MODEL")
    rerank_model: str | None = Field(
        default=None,
        validation_alias="INFERENCE_RERANK_MODEL",
    )
    chat_endpoint: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="INFERENCE_CHAT_ENDPOINT",
    )
    embedding_endpoint: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="INFERENCE_EMBEDDING_ENDPOINT",
    )
    rerank_endpoint: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="INFERENCE_RERANK_ENDPOINT",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_api_key_file: str | None = Field(default=None, alias="OPENROUTER_API_KEY_FILE")
    phoenix_tracing_enabled: bool = Field(default=False, alias="PHOENIX_TRACING_ENABLED")
    phoenix_collector_endpoint: str = Field(
        default="https://phoenix-otlp.docker.home.arpa/v1/traces",
        alias="PHOENIX_COLLECTOR_ENDPOINT",
    )
    phoenix_project_name: str = Field(default="camillo", alias="PHOENIX_PROJECT_NAME")
    decay_rate: float = Field(alias="DECAY_RATE")
    recall_top_k: int = Field(alias="RECALL_TOP_K")
    recall_vector_limit: int = Field(alias="RECALL_VECTOR_LIMIT")
    recall_full_text_search_limit: int = Field(alias="RECALL_FULL_TEXT_SEARCH_LIMIT")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_min_score: float = Field(default=0.35, alias="RERANK_MIN_SCORE")
    rrf_k: int = Field(default=60, alias="RRF_K")
    recall_candidate_limit: int = Field(default=30, alias="RECALL_CANDIDATE_LIMIT")
    dreaming_enabled: bool = Field(default=True, alias="DREAMING_ENABLED")
    dreaming_interval_seconds: int = Field(default=900, alias="DREAMING_INTERVAL_SECONDS")
    dreaming_dry_run: bool = Field(default=False, alias="DREAMING_DRY_RUN")
    dreaming_seed_limit: int = Field(default=5, alias="DREAMING_SEED_LIMIT")
    dreaming_batch_size: int = Field(default=5, alias="DREAMING_BATCH_SIZE")
    dreaming_min_similarity: float = Field(default=0.75, alias="DREAMING_MIN_SIMILARITY")
    dreaming_min_synthesis_confidence: float = Field(
        default=0.6,
        alias="DREAMING_MIN_SYNTHESIS_CONFIDENCE",
    )
    dreaming_model: str | None = Field(
        default=None,
        alias="INFERENCE_DREAM_MODEL",
    )
    dreaming_endpoint: str | None = Field(
        default=None,
        alias="INFERENCE_DREAM_ENDPOINT",
    )
    chat_temperature: float | None = Field(default=None, alias="INFERENCE_CHAT_TEMPERATURE")
    chat_reasoning_effort: str | None = Field(default=None, alias="INFERENCE_CHAT_REASONING_EFFORT")
    chat_max_tokens: int | None = Field(default=None, alias="INFERENCE_CHAT_MAX_TOKENS")
    chat_extra_kwargs: dict[str, object] | None = Field(
        default=None, alias="INFERENCE_CHAT_EXTRA_KWARGS"
    )
    embedding_extra_kwargs: dict[str, object] | None = Field(
        default=None, alias="INFERENCE_EMBEDDING_EXTRA_KWARGS"
    )
    rerank_extra_kwargs: dict[str, object] | None = Field(
        default=None, alias="INFERENCE_RERANK_EXTRA_KWARGS"
    )
    dream_temperature: float | None = Field(default=None, alias="INFERENCE_DREAM_TEMPERATURE")
    dream_reasoning_effort: str | None = Field(
        default=None, alias="INFERENCE_DREAM_REASONING_EFFORT"
    )
    dream_max_tokens: int | None = Field(default=None, alias="INFERENCE_DREAM_MAX_TOKENS")
    dream_extra_kwargs: dict[str, object] | None = Field(
        default=None, alias="INFERENCE_DREAM_EXTRA_KWARGS"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_ids(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        for field in ("chat_model", "embedding_model", "rerank_model", "dreaming_model"):
            if field in values and isinstance(values[field], str):
                values[field] = values[field].removeprefix("openrouter/").removeprefix("openai/")
        if not values.get("openrouter_api_key") and values.get("openrouter_api_key_file"):
            values["openrouter_api_key"] = _read_secret_file(values["openrouter_api_key_file"])
        return values

    @model_validator(mode="after")
    def build_database_url(self) -> Settings:
        """Prefer secret-backed Postgres parts while preserving DATABASE_URL overrides.

        Returns:
            The settings instance with a usable SQLAlchemy URL.

        Raises:
            ValueError: If neither a full URL nor a Postgres password source is configured.
        """
        if self.database_url:
            return self

        password = self.postgres_password
        if not password and self.postgres_password_file:
            password = _read_secret_file(self.postgres_password_file)
        if not password:
            raise ValueError("DATABASE_URL or POSTGRES_PASSWORD_FILE must be configured")

        user = quote(self.postgres_user, safe="")
        escaped_password = quote(password, safe="")
        database = quote(self.postgres_db, safe="")
        self.database_url = (
            f"postgresql+asyncpg://{user}:{escaped_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{database}"
        )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for process-wide configuration."""
    return Settings()


settings = get_settings()
