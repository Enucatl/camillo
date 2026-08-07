from camillo.cognitive.cognitive_math import calculate_activation
from camillo.cognitive.recall_utils import Candidate, normalize_scores, reciprocal_rank_fusion
from camillo.interfaces import EmbeddingProvider, MemoryStoreProtocol
from camillo.settings import settings


class RecallService:
    """Run hybrid corpus retrieval with fixed, explainable ranking weights."""

    def __init__(self, memory_store: MemoryStoreProtocol, llm_service: EmbeddingProvider):
        """Accept storage and provider ports so ranking remains unit-testable."""
        self.memory_store = memory_store
        self.llm_service = llm_service

    async def recall(self, query: str, top_k: int, workspace: str | None = None) -> list[Candidate]:
        """Retrieve memories and update access bookkeeping for public recall."""
        candidates = await self.search(query, top_k, workspace)
        await self.memory_store.mark_accessed([candidate.memory.id for candidate in candidates])
        return candidates

    async def search(self, query: str, top_k: int, workspace: str | None = None) -> list[Candidate]:
        """Retrieve without mutation for deduplication, replacement, and dreaming."""
        embedding = await self.llm_service.get_embedding(query)
        vector = await self.memory_store.vector_candidates(embedding, settings.recall_vector_limit)
        lexical = await self.memory_store.full_text_search_candidates(
            query, settings.recall_full_text_search_limit
        )
        candidates = reciprocal_rank_fusion(
            vector, lexical, rrf_k=settings.rrf_k, limit=settings.recall_candidate_limit
        )
        normalized = normalize_scores([candidate.rrf_score or 0.0 for candidate in candidates])
        for candidate, score in zip(candidates, normalized, strict=True):
            candidate.rrf_score = score
        if settings.rerank_enabled and candidates:
            scores = normalize_scores(
                await self.llm_service.rerank_results(
                    query, [candidate.memory.raw_content for candidate in candidates]
                )
            )
            for candidate, score in zip(candidates, scores, strict=False):
                candidate.rerank_score = score
        for candidate in candidates:
            activation = calculate_activation(
                candidate.memory.base_importance,
                candidate.memory.access_count,
                candidate.memory.last_accessed_at,
                decay_rate=settings.decay_rate,
            )
            affinity = 1.0 if workspace and candidate.memory.workspace == workspace else 0.0
            candidate.activation_score = activation
            candidate.workspace_affinity_score = affinity
            relevance = candidate.retrieval_score
            activation_score = min(activation / 1.5, 1.0)
            candidate.final_score = 0.75 * relevance + 0.15 * activation_score + 0.10 * affinity
        candidates.sort(key=lambda candidate: candidate.final_score or 0.0, reverse=True)
        return candidates[:top_k]
