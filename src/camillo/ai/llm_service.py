import litellm
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
        """Use reranking as an optional relevance signal, not a hard dependency.

        Recall should remain available when no rerank model is configured or a
        provider returns a shape LiteLLM does not normalize consistently.

        Args:
            query: The recall query.
            documents: Candidate memory texts.

        Returns:
            One relevance score per document, with order-preserving fallback.
        """
        if not documents:
            return []

        fallback = [1.0 - (index / max(len(documents), 1)) * 0.2 for index in range(len(documents))]
        if not settings.litellm_rerank_model:
            return fallback

        try:
            response = await litellm.arerank(
                model=settings.litellm_rerank_model,
                query=query,
                documents=documents,
            )
            results = _response_value(response, "results") or []
            scores = [0.0] * len(documents)

            for item in results:
                index = _response_value(item, "index")
                score = _response_value(item, "relevance_score")
                if score is None:
                    score = _response_value(item, "score")
                if index is None:
                    continue
                index = int(index)
                if 0 <= index < len(scores):
                    scores[index] = float(score or 0.0)

            return scores
        except Exception:
            logger.exception("Failed to rerank recall candidates; using fallback")
            return fallback


def _response_value(item: object, key: str) -> object | None:
    """Handle provider response shape drift behind one defensive accessor.

    Args:
        item: Dict-like or attribute-based LiteLLM response object.
        key: Field name to read.

    Returns:
        The field value when present, otherwise `None`.
    """
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
