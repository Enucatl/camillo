import json
import re

import litellm
from loguru import logger

from camillo.interfaces import EmbeddingProvider, Reranker
from camillo.settings import settings

OPENROUTER_RERANK_API_BASE = "https://openrouter.ai/api/v1/rerank"


class LiteLLMService(EmbeddingProvider, Reranker):
    """LiteLLM adapter for embeddings, optional reranking, and dreaming."""

    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding for safe, already-redacted text."""
        response = await litellm.aembedding(model=settings.litellm_embedding_model, input=[text])
        return [float(value) for value in response.data[0]["embedding"]]

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Return provider relevance scores, falling back to stable rank order."""
        if not documents:
            return []
        fallback = [1.0 - index / max(len(documents), 1) * 0.2 for index in range(len(documents))]
        if not settings.litellm_rerank_model:
            return fallback
        try:
            rerank_kwargs = _rerank_provider_kwargs(settings.litellm_rerank_model)
            response = await litellm.arerank(
                model=rerank_kwargs.pop("model"),
                query=query,
                documents=documents,
                **rerank_kwargs,
            )
            scores = [0.0] * len(documents)
            for item in response.results:
                index = int(_value(item, "index") or -1)
                if 0 <= index < len(scores):
                    scores[index] = float(
                        _value(item, "relevance_score") or _value(item, "score") or 0.0
                    )
            return scores
        except Exception:
            logger.exception("Reranking failed; using fallback order")
            return fallback

    async def synthesize_dream(self, cluster_memories: list[str]) -> dict[str, object]:
        """Request at most one durable proposal from a qualifying episode batch."""
        if len(cluster_memories) < 2:
            return {"content": None, "memory_type": "fact", "confidence": 0.0}
        prompt = (
            "Synthesize at most one durable fact from these episodes. Return JSON with "
            "content, memory_type, confidence.\n" + "\n".join(cluster_memories)
        )
        try:
            response = await litellm.acompletion(
                model=settings.dreaming_model or settings.litellm_completion_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            parsed = json.loads(_strip_json_fence(response.choices[0].message.content or "{}"))
            return parsed if isinstance(parsed, dict) else {"content": None, "confidence": 0.0}
        except Exception:
            logger.exception("Dream synthesis failed")
            return {"content": None, "confidence": 0.0}


def _rerank_provider_kwargs(model: str) -> dict[str, str]:
    """Route OpenRouter rerank models through its Cohere-compatible endpoint."""
    if settings.openrouter_api_key and (
        model.startswith("openrouter/") or model.startswith("cohere/rerank")
    ):
        return {
            "model": model.removeprefix("openrouter/"),
            "custom_llm_provider": "litellm_proxy",
            "api_base": OPENROUTER_RERANK_API_BASE,
            "api_key": settings.openrouter_api_key,
        }
    return {"model": model}


def _value(item: object, key: str) -> object | None:
    """Read a field from dict-like or attribute-based provider responses."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _strip_json_fence(content: str) -> str:
    """Remove an optional markdown JSON fence from provider output."""
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else content.strip()
