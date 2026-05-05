import json

import litellm
from loguru import logger

from camillo.ai.prompts import render_valence_prompt
from camillo.db.models import Memory
from camillo.interfaces import CompletionProvider, EmbeddingProvider, Reranker
from camillo.schemas.submit_memory import MemoryRelationshipClassification
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
            response = await litellm.acompletion(
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
        response = await litellm.aembedding(model=settings.litellm_embedding_model, input=[text])
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

    async def classify_memory_relationships(
        self,
        intent: str,
        new_content: str,
        existing_memories: list[Memory],
    ) -> list[MemoryRelationshipClassification]:
        """Classify new durable memory against related active memories.

        Args:
            intent: Caller intent passed through to the classifier.
            new_content: Proposed durable memory text.
            existing_memories: Recalled memories to compare with the candidate.

        Returns:
            One defensive classification per existing memory.
        """
        if not existing_memories:
            return []

        fallback = _fallback_relationships(len(existing_memories))
        numbered_memories = "\n".join(
            f"{index}. {memory.raw_content}" for index, memory in enumerate(existing_memories)
        )
        prompt = _render_relationship_prompt(intent, new_content, numbered_memories)

        try:
            response = await litellm.acompletion(
                model=settings.litellm_completion_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content.strip())
            if not isinstance(parsed, list):
                return fallback

            classifications_by_index: dict[int, MemoryRelationshipClassification] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                raw_index = item.get("index")
                if (
                    not isinstance(raw_index, int)
                    or raw_index < 0
                    or raw_index >= len(existing_memories)
                ):
                    continue
                if "confidence" in item:
                    item["confidence"] = max(0.0, min(float(item["confidence"]), 1.0))
                classification = MemoryRelationshipClassification.model_validate(item)
                classifications_by_index[classification.index] = classification

            return [
                classifications_by_index.get(index, fallback[index])
                for index in range(len(existing_memories))
            ]
        except Exception:
            logger.exception("Failed to classify memory relationships; using fallback")
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


def _fallback_relationships(count: int) -> list[MemoryRelationshipClassification]:
    """Build index-complete unavailable classifications.

    Args:
        count: Number of existing memories being classified.

    Returns:
        Unrelated classifications that keep reconciliation conservative.
    """
    return [
        MemoryRelationshipClassification(
            index=index,
            relation="unrelated",
            confidence=0.0,
            contradiction_type="none",
            resolution="keep_both",
            rationale="classification unavailable",
            old_memory_refinement=None,
            new_memory_refinement=None,
        )
        for index in range(count)
    ]


def _render_relationship_prompt(
    intent: str,
    new_content: str,
    numbered_memories: str,
) -> str:
    """Render the contradiction-aware classifier prompt.

    Args:
        intent: Caller intent for the memory submission.
        new_content: Candidate memory.
        numbered_memories: Existing memories as an indexed list.

    Returns:
        Prompt asking the model for strict JSON.
    """
    return f"""You are reconciling a new memory candidate against existing memories.

Your job is not only to detect contradiction.
Diagnose whether an apparent contradiction can be resolved by context, time,
scope, environment, or specificity.

Classify how the new content relates to each existing memory.

Allowed relation labels:
- confirms
- extends
- contradicts
- supersedes
- forgets
- unrelated
- duplicate

Allowed contradiction_type values:
- none
- direct_conflict
- temporal_shift
- context_shift
- scope_mismatch
- environment_difference
- preference_change
- implementation_change
- ambiguous

Allowed resolution values:
- keep_both
- supersede_old
- deprecate_old
- refine_old
- create_exception
- needs_review

Important rules:
- Do not mark an old memory superseded merely because there is a surface-level conflict.
- Prefer keep_both, refine_old, or create_exception when the conflict may be contextual.
- Use supersede_old only when the new content clearly says the old memory is
  outdated, replaced, or wrong.
- Use deprecate_old when the new content explicitly says to forget, stop using,
  or invalidate the old memory.
- Preserve both memories when they can be made true with contextual qualifications.

Intent: {intent}

New content:
{new_content}

Existing memories:
{numbered_memories}

Return only valid JSON in this shape:
[
  {{
    "index": 0,
    "relation": "contradicts",
    "confidence": 0.86,
    "contradiction_type": "environment_difference",
    "resolution": "keep_both",
    "rationale": "The memories can both be true in different environments.",
    "old_memory_refinement": "Redis is used for production caching.",
    "new_memory_refinement": "Local tests use in-memory caching."
  }}
]"""
