from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any
from uuid import UUID

from camillo.db.models import Memory


@dataclass
class Candidate:
    """Carry scoring provenance so recall decisions stay explainable.

    Phase 2 blends retrieval, reranking, activation, and graph context. Keeping
    those values together prevents the API from collapsing the pipeline into an
    opaque score while still preserving the old top-level score contract.
    """

    memory: Memory
    vector_score: float | None = None
    text_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    activation_score: float | None = None
    scope_affinity_score: float | None = None
    final_score: float | None = None
    source: str = "primary"
    linked_from: UUID | None = None
    edge_weight: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_score(self) -> float:
        """Prefer the latest relevance signal without losing fallback behavior.

        Returns:
            The score downstream ranking should treat as direct retrieval
            relevance, falling back through the available Phase 1/2 signals.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        scores = [score for score in (self.vector_score, self.text_score) if score is not None]
        return max(scores) if scores else 0.0


def reciprocal_rank_fusion(
    vector_results: list[tuple[Memory, float]],
    text_results: list[tuple[Memory, float]],
    *,
    rrf_k: int,
    limit: int,
) -> list[Candidate]:
    """Blend vector and lexical rankings without assuming comparable scales.

    RRF uses rank position instead of raw scores so pgvector similarity and
    trigram scores can contribute fairly before the reranker sees candidates.

    Args:
        vector_results: Ranked vector matches with provider-specific scores.
        text_results: Ranked lexical matches with database-specific scores.
        rrf_k: Dampening constant that controls how steeply rank contributes.
        limit: Maximum number of fused candidates to return.

    Returns:
        Candidates sorted by fused retrieval strength.
    """
    merged: dict[UUID, Candidate] = {}

    def add_results(results: list[tuple[Memory, float]], attr_name: str) -> None:
        """Accumulate one ranked source while preserving source-specific scores.

        Args:
            results: Ordered memory/score pairs from a retrieval backend.
            attr_name: Candidate attribute that should receive the raw score.
        """
        for rank, (memory, score) in enumerate(results, start=1):
            candidate = merged.get(memory.id)
            if candidate is None:
                candidate = Candidate(memory=memory)
                merged[memory.id] = candidate

            current = getattr(candidate, attr_name)
            if current is None or score > current:
                setattr(candidate, attr_name, score)

            candidate.rrf_score = (candidate.rrf_score or 0.0) + 1.0 / (rrf_k + rank)

    add_results(vector_results, "vector_score")
    add_results(text_results, "text_score")

    candidates = list(merged.values())
    candidates.sort(key=lambda candidate: candidate.rrf_score or 0.0, reverse=True)
    return candidates[:limit]


def normalize_scores(values: list[float]) -> list[float]:
    """Put heterogeneous provider scores on a stable comparison scale.

    Args:
        values: Scores from one ranking source.

    Returns:
        Scores normalized into `[0.0, 1.0]`, preserving useful all-equal
        positive signals as `1.0` instead of erasing them.
    """
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        value = 1.0 if maximum > 0 else 0.0
        return [value for _ in values]

    span = maximum - minimum
    return [(value - minimum) / span for value in values]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compare embeddings without adding a heavy numeric dependency.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity, or `0.0` when either vector cannot carry direction.
    """
    dot = sum(left * right for left, right in zip(a, b, strict=False))
    norm_a = sqrt(sum(value * value for value in a))
    norm_b = sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_to_list(value: Any) -> list[float] | None:
    """Normalize database vectors before pure-Python similarity checks.

    Args:
        value: A pgvector, list-like object, or `None`.

    Returns:
        A plain float list so diversity logic does not depend on storage type.
    """
    if value is None:
        return None
    return [float(item) for item in value]


def apply_diversity_filter(
    candidates: list[Candidate],
    *,
    similarity_threshold: float,
    max_results: int,
) -> list[Candidate]:
    """Preserve result variety after scoring has already found relevance.

    The recall response should not spend its small top-K budget on near-clones
    when another relevant memory can add context.

    Args:
        candidates: Final-score-sorted primary candidates.
        similarity_threshold: Cosine value at which a candidate is too similar.
        max_results: Maximum number of diverse candidates to keep.

    Returns:
        A score-ordered subset with near-duplicate embeddings removed.
    """
    kept: list[Candidate] = []

    for candidate in candidates:
        candidate_embedding = embedding_to_list(candidate.memory.embedding)
        too_similar = False

        for existing in kept:
            existing_embedding = embedding_to_list(existing.memory.embedding)
            if candidate_embedding is None or existing_embedding is None:
                continue

            if cosine_similarity(candidate_embedding, existing_embedding) >= similarity_threshold:
                too_similar = True
                break

        if not too_similar:
            kept.append(candidate)

        if len(kept) >= max_results:
            break

    return kept
