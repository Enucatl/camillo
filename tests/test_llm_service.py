from types import SimpleNamespace

import pytest

from camillo.ai import llm_service
from camillo.ai.llm_service import LiteLLMService


@pytest.mark.asyncio
async def test_score_valence_parses_float(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect valence parsing so provider strings become numeric memory weights."""

    async def fake_acompletion(**_kwargs):
        """Return parseable content so the test isolates conversion logic."""
        message = SimpleNamespace(content="0.82")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_service, "acompletion", fake_acompletion)

    assert await LiteLLMService().score_valence("important") == 0.82


@pytest.mark.asyncio
async def test_score_valence_falls_back_on_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect ingestion from malformed provider output."""

    async def fake_acompletion(**_kwargs):
        """Return invalid content so fallback behavior is deterministic."""
        message = SimpleNamespace(content="not-a-number")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_service, "acompletion", fake_acompletion)

    assert await LiteLLMService().score_valence("temporary chatter") == 0.5


@pytest.mark.asyncio
async def test_get_embedding_returns_first_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect embedding conversion to plain floats for pgvector storage."""

    async def fake_aembedding(**_kwargs):
        """Return a LiteLLM-shaped embedding response without a network call."""
        return SimpleNamespace(data=[{"embedding": [1, 2.5, -3]}])

    monkeypatch.setattr(llm_service, "aembedding", fake_aembedding)

    assert await LiteLLMService().get_embedding("query") == [1.0, 2.5, -3.0]


@pytest.mark.asyncio
async def test_rerank_fallback_scores_preserve_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect fallback reranking from changing candidate order semantics."""
    monkeypatch.setattr(llm_service.settings, "litellm_rerank_model", None)

    scores = await LiteLLMService().rerank_results("query", ["short", "a much longer document"])

    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.9)
