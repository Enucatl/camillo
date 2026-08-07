import json
import re
from functools import lru_cache

import niquests
from loguru import logger
from shared_inference import InferenceClient

from camillo.interfaces import EmbeddingProvider, Reranker
from camillo.settings import settings


def _provider_for(endpoint: str) -> str:
    return "openrouter" if "openrouter.ai" in endpoint else "openai-compatible"


def _key_for(endpoint: str) -> str | None:
    return settings.openrouter_api_key if _provider_for(endpoint) == "openrouter" else None


class InferenceService(EmbeddingProvider, Reranker):
    """Shared inference adapter for embeddings, reranking, and dreaming."""

    def __init__(self) -> None:
        self._session = niquests.AsyncSession(timeout=120)
        self._chat = InferenceClient(
            base_url=settings.chat_endpoint,
            api_key=_key_for(settings.chat_endpoint),
            provider=_provider_for(settings.chat_endpoint),
            domain="camillo",
            session=self._session,
        )
        self._embedding = InferenceClient(
            base_url=settings.embedding_endpoint,
            api_key=_key_for(settings.embedding_endpoint),
            provider=_provider_for(settings.embedding_endpoint),
            domain="camillo",
            session=self._session,
        )
        self._rerank = InferenceClient(
            base_url=settings.rerank_endpoint,
            api_key=_key_for(settings.rerank_endpoint),
            provider=_provider_for(settings.rerank_endpoint),
            domain="camillo",
            session=self._session,
        )

    async def close(self) -> None:
        """Close the shared provider session during application shutdown."""
        await self._session.close()

    async def get_embedding(self, text: str, *, domain: str = "document_embedding") -> list[float]:
        """Generate an embedding for safe, already-redacted text."""
        response = await self._embedding.embed(
            model=settings.embedding_model, input=[text], domain=domain
        )
        return response.embeddings[0]

    async def rerank_results(
        self, query: str, documents: list[str], *, domain: str = "recall_rerank"
    ) -> list[float]:
        """Return provider relevance scores, falling back to stable rank order."""
        if not documents:
            return []
        fallback = [1.0 - index / max(len(documents), 1) * 0.2 for index in range(len(documents))]
        if not settings.rerank_model:
            return fallback
        try:
            model = settings.rerank_model.removeprefix("openrouter/")
            response = await self._rerank.rerank(
                model=model, query=query, documents=documents, domain=domain
            )
            scores = [0.0] * len(documents)
            for index, score in zip(response.indices, response.scores, strict=False):
                if 0 <= index < len(scores):
                    scores[index] = score
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
            response = await self._chat.complete(
                model=settings.dreaming_model or settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                domain="dream_consolidation",
                temperature=0,
            )
            parsed = json.loads(_strip_json_fence(response.content or "{}"))
            return parsed if isinstance(parsed, dict) else {"content": None, "confidence": 0.0}
        except Exception:
            logger.exception("Dream synthesis failed")
            return {"content": None, "confidence": 0.0}


def _strip_json_fence(content: str) -> str:
    """Remove an optional markdown JSON fence from provider output."""
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else content.strip()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return the process-wide provider and its long-lived session."""
    return InferenceService()
