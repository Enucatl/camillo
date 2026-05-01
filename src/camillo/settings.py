from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralize environment parsing so recall tuning is operationally safe.

    The service needs Phase 2 behavior to be configurable without code changes,
    while keeping validation close to the values that can break ranking.
    """

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
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_min_score: float = Field(default=0.35, alias="RERANK_MIN_SCORE")
    rrf_k: int = Field(default=60, alias="RRF_K")
    recall_candidate_limit: int = Field(default=30, alias="RECALL_CANDIDATE_LIMIT")
    recall_relevance_weight: float = Field(default=0.7, alias="RECALL_RELEVANCE_WEIGHT")
    recall_activation_weight: float = Field(default=0.3, alias="RECALL_ACTIVATION_WEIGHT")
    diversity_enabled: bool = Field(default=True, alias="DIVERSITY_ENABLED")
    diversity_similarity_threshold: float = Field(
        default=0.92,
        alias="DIVERSITY_SIMILARITY_THRESHOLD",
    )
    hebbian_spread_enabled: bool = Field(default=True, alias="HEBBIAN_SPREAD_ENABLED")
    hebbian_spread_limit: int = Field(default=3, alias="HEBBIAN_SPREAD_LIMIT")
    hebbian_max_depth: int = Field(default=1, alias="HEBBIAN_MAX_DEPTH")
    hebbian_edge_threshold: float = Field(alias="HEBBIAN_EDGE_THRESHOLD")
    reinforcement_enabled: bool = Field(default=True, alias="REINFORCEMENT_ENABLED")
    reinforcement_edge_increment: float = Field(
        default=1.0,
        alias="REINFORCEMENT_EDGE_INCREMENT",
    )

    @model_validator(mode="after")
    def validate_recall_weights(self) -> "Settings":
        """Prevent unusable ranking weights before the app starts.

        Returns:
            The validated settings instance for Pydantic's model pipeline.

        Raises:
            ValueError: If relevance and activation weights cannot be normalized.
        """
        total = self.recall_relevance_weight + self.recall_activation_weight
        if total <= 0:
            raise ValueError("Recall weights must sum to a positive value")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for process-wide configuration."""
    return Settings()


settings = get_settings()
