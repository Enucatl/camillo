import logging

from camillo.settings import settings

logger = logging.getLogger(__name__)


def warn_missing_provider_keys() -> None:
    """Warn when an explicitly configured OpenRouter route lacks its key."""
    configured_models = (
        settings.chat_model,
        settings.embedding_model,
        settings.rerank_model or "",
    )
    uses_openrouter = any(model.startswith("openrouter/") for model in configured_models)

    if uses_openrouter and not settings.openrouter_api_key:
        logger.warning(
            "OpenRouter models are configured but OPENROUTER_API_KEY is unset. "
            "Health checks will still pass, but /ingest and /recall LLM calls will fail "
            "until the key is provided."
        )
