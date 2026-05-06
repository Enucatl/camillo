from types import SimpleNamespace

import pytest

from camillo.ai import llm_service
from camillo.ai.llm_service import LiteLLMService


@pytest.mark.asyncio
async def test_get_embedding_returns_first_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect embedding conversion to plain floats for pgvector storage."""

    async def fake_aembedding(**_kwargs: object) -> SimpleNamespace:
        """Return a LiteLLM-shaped embedding response without a network call."""
        return SimpleNamespace(data=[{"embedding": [1, 2.5, -3]}])

    monkeypatch.setattr(llm_service.litellm, "aembedding", fake_aembedding)

    assert await LiteLLMService().get_embedding("query") == [1.0, 2.5, -3.0]


@pytest.mark.asyncio
async def test_rerank_fallback_scores_preserve_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect fallback reranking from changing candidate order semantics."""
    monkeypatch.setattr(llm_service.settings, "litellm_rerank_model", None)

    scores = await LiteLLMService().rerank_results("query", ["short", "a much longer document"])

    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("configured_model", "provider_model"),
    [
        ("openrouter/cohere/rerank-4-pro", "cohere/rerank-4-pro"),
        ("cohere/rerank-4-pro", "cohere/rerank-4-pro"),
    ],
)
@pytest.mark.asyncio
async def test_rerank_routes_openrouter_models_to_openrouter_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    configured_model: str,
    provider_model: str,
) -> None:
    """Protect OpenRouter credit routing for rerank models LiteLLM parses as Cohere."""
    monkeypatch.setattr(llm_service.settings, "litellm_rerank_model", configured_model)
    monkeypatch.setattr(llm_service.settings, "openrouter_api_key", "test-key")
    call_kwargs = {}

    async def fake_arerank(**kwargs: object) -> SimpleNamespace:
        """Capture LiteLLM rerank routing without making a network call."""
        call_kwargs.update(kwargs)
        return SimpleNamespace(results=[{"index": 1, "relevance_score": 0.8}])

    monkeypatch.setattr(llm_service.litellm, "arerank", fake_arerank)

    scores = await LiteLLMService().rerank_results("query", ["short", "a much longer document"])

    assert call_kwargs["model"] == provider_model
    assert call_kwargs["custom_llm_provider"] == "litellm_proxy"
    assert call_kwargs["api_base"] == llm_service.OPENROUTER_RERANK_API_BASE
    assert call_kwargs["api_key"] == "test-key"
    assert scores == [0.0, 0.8]
