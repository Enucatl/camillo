from litellm import acompletion, aembedding
from loguru import logger

from camillo.ai.prompts import render_valence_prompt
from camillo.interfaces import CompletionProvider, EmbeddingProvider, Reranker
from camillo.settings import settings


class LiteLLMService(CompletionProvider, EmbeddingProvider, Reranker):
    """LiteLLM-backed implementation of the AI provider interfaces."""

    async def score_valence(self, raw_content: str) -> float:
        """Score whether an interaction is worth retaining long-term.

        Args:
            raw_content: The conversation content to classify.

        Returns:
            A clamped continuous score from 0.0 to 1.0, with 0.5 as a neutral
            fallback when the provider response cannot be parsed.
        """
        try:
            response = await acompletion(
                model=settings.litellm_completion_model,
                messages=[{"role": "user", "content": render_valence_prompt(raw_content)}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            score = float(content.strip())
        except Exception:
            logger.exception("Failed to score memory valence; using default")
            return 0.5

        return max(0.0, min(score, 1.0))

    async def get_embedding(self, text: str) -> list[float]:
        """Embed text using the configured LiteLLM embedding model.

        Args:
            text: The text to embed.

        Returns:
            A list of floats matching the configured embedding dimension.
        """
        response = await aembedding(model=settings.litellm_embedding_model, input=[text])
        embedding = response.data[0]["embedding"]
        return [float(value) for value in embedding]

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Rank candidate documents for a query.

        Args:
            query: The recall query.
            documents: Candidate memory texts.

        Returns:
            Placeholder relevance scores until Phase 2 wires a real reranker.
        """
        # TODO(Phase 2): route through LiteLLM reranking when the provider is configured.
        if not documents:
            return []
        max_len = max(len(document) for document in documents) or 1
        return [max(0.1, min(len(document) / max_len, 1.0)) for document in documents]
