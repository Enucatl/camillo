from pathlib import Path

from camillo.settings import Settings


def _settings_kwargs(**overrides: str | None) -> dict[str, str | None]:
    """Create explicit settings input so tests do not depend on host env state.

    Args:
        overrides: Settings values to replace in the baseline.

    Returns:
        A complete Settings input dictionary keyed by environment aliases.
    """
    values: dict[str, str | None] = {
        "APP_NAME": "Camillo",
        "APP_ENV": "test",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "",
        "POSTGRES_USER": "camillo",
        "POSTGRES_PASSWORD": None,
        "POSTGRES_PASSWORD_FILE": None,
        "POSTGRES_DB": "camillo",
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": "5432",
        "EMBEDDING_DIM": "1024",
        "LITELLM_COMPLETION_MODEL": "openrouter/google/gemma-4-31b-it:free",
        "LITELLM_EMBEDDING_MODEL": "openrouter/baai/bge-m3",
        "LITELLM_RERANK_MODEL": "openrouter/cohere/rerank-4-pro",
        "OPENROUTER_API_KEY": "",
        "PHOENIX_TRACING_ENABLED": "false",
        "PHOENIX_COLLECTOR_ENDPOINT": "https://phoenix-otlp.docker.home.arpa/v1/traces",
        "PHOENIX_PROJECT_NAME": "camillo-test",
        "DECAY_RATE": "0.01",
        "RECALL_TOP_K": "5",
        "RECALL_VECTOR_LIMIT": "30",
        "RECALL_FULL_TEXT_SEARCH_LIMIT": "30",
        "HEBBIAN_EDGE_THRESHOLD": "2.0",
    }
    values.update(overrides)
    return values


def test_builds_database_url_from_secret_file(tmp_path: Path) -> None:
    """Protect Compose secret support from regressing to password-bearing env URLs."""
    secret_file = tmp_path / "postgres_password"
    secret_file.write_text("s3cr%t\n", encoding="utf-8")

    settings = Settings(**_settings_kwargs(POSTGRES_PASSWORD_FILE=str(secret_file)))

    assert settings.database_url == "postgresql+asyncpg://camillo:s3cr%25t@postgres:5432/camillo"


def test_database_url_override_remains_supported(tmp_path: Path) -> None:
    """Keep one-shot operational overrides available for non-Compose workflows."""
    secret_file = tmp_path / "postgres_password"
    secret_file.write_text("ignored", encoding="utf-8")

    settings = Settings(
        **_settings_kwargs(
            DATABASE_URL="postgresql+asyncpg://override:override@db:5432/override",
            POSTGRES_PASSWORD_FILE=str(secret_file),
        )
    )

    assert settings.database_url == "postgresql+asyncpg://override:override@db:5432/override"
