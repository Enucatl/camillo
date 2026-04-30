import logging

from camillo.settings import settings

logger = logging.getLogger(__name__)
_configured = False


def configure_phoenix_tracing() -> bool:
    """Configure Phoenix tracing once when the optional trace extra is installed."""
    global _configured

    if _configured:
        return True
    if not settings.phoenix_tracing_enabled:
        return False

    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from phoenix.otel import register
    except ImportError:
        logger.exception(
            "Phoenix tracing is enabled but tracing packages are not installed. "
            "Install the 'trace' extra."
        )
        return False

    register(
        endpoint=settings.phoenix_collector_endpoint,
        project_name=settings.phoenix_project_name,
        auto_instrument=False,
    )
    LiteLLMInstrumentor().instrument()
    _configured = True
    logger.info(
        "Phoenix tracing configured for project '%s' at %s",
        settings.phoenix_project_name,
        settings.phoenix_collector_endpoint,
    )
    return True
