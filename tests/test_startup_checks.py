import logging

from camillo import startup_checks


def test_warn_missing_openrouter_key_when_openrouter_models_are_configured(
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        startup_checks.settings,
        "litellm_completion_model",
        "openrouter/google/gemma-4-31b-it:free",
    )
    monkeypatch.setattr(
        startup_checks.settings, "litellm_embedding_model", "openrouter/baai/bge-m3"
    )
    monkeypatch.setattr(
        startup_checks.settings, "litellm_rerank_model", "openrouter/cohere/rerank-4-pro"
    )
    monkeypatch.setattr(startup_checks.settings, "openrouter_api_key", "")

    with caplog.at_level(logging.WARNING):
        startup_checks.warn_missing_provider_keys()

    assert "OPENROUTER_API_KEY is unset" in caplog.text


def test_no_warning_when_openrouter_key_is_configured(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        startup_checks.settings,
        "litellm_completion_model",
        "openrouter/google/gemma-4-31b-it:free",
    )
    monkeypatch.setattr(
        startup_checks.settings, "litellm_embedding_model", "openrouter/baai/bge-m3"
    )
    monkeypatch.setattr(
        startup_checks.settings, "litellm_rerank_model", "openrouter/cohere/rerank-4-pro"
    )
    monkeypatch.setattr(startup_checks.settings, "openrouter_api_key", "test-key")

    with caplog.at_level(logging.WARNING):
        startup_checks.warn_missing_provider_keys()

    assert "OPENROUTER_API_KEY is unset" not in caplog.text
