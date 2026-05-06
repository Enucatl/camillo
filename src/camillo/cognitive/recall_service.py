from uuid import UUID

from camillo.cognitive.cognitive_math import calculate_activation
from camillo.cognitive.recall_utils import (
    Candidate,
    apply_diversity_filter,
    normalize_scores,
    reciprocal_rank_fusion,
)
from camillo.cognitive.scope_policy import scope_affinity
from camillo.interfaces import EmbeddingProvider, GraphStoreProtocol, MemoryStoreProtocol
from camillo.settings import settings


class RecallService:
    """Keep recall orchestration explicit as cognitive behavior grows.

    Phase 2 has enough steps that hiding them behind one block would make future
    ranking changes risky. The service keeps each stage isolated so retrieval,
    graph spreading, and reinforcement can evolve independently.
    """

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        graph_store: GraphStoreProtocol,
        llm_service: EmbeddingProvider,
    ):
        """Accept ports so the pipeline can be tested without PostgreSQL or LiteLLM.

        Args:
            memory_store: Persistence boundary for direct memory retrieval.
            graph_store: Persistence boundary for Hebbian associations.
            llm_service: Provider used for embeddings and reranking.
        """
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.llm_service = llm_service

    async def recall(
        self,
        namespace: str,
        query: str,
        top_k: int,
        *,
        include_hebbian: bool = True,
        include_shared: bool = True,
    ) -> list[Candidate]:
        """Run the full recall path while preserving primary-result priority.

        Hebbian memories are appended after direct matches because graph context
        should enrich recall, not displace the answer most relevant to the query.

        Args:
            namespace: Memory partition to query.
            query: Natural-language recall prompt.
            top_k: Number of primary memories to return before graph expansion.
            include_hebbian: Whether graph-linked memories may be appended.
            include_shared: Whether shared/global cross-namespace memories are
                eligible for direct recall.

        Returns:
            Primary candidates followed by optional Hebbian candidates.
        """
        query_embedding = await self.llm_service.get_embedding(query)
        candidates = await self._generate_candidates(
            namespace,
            query,
            query_embedding,
            include_shared=include_shared,
        )
        if not candidates:
            return []

        candidates = await self._rerank_candidates(query, candidates)
        candidates = self._apply_relevance_threshold(candidates)
        candidates = self._score_activation_and_final(candidates, namespace)
        primary = self._select_primary(candidates, top_k)
        hebbian = await self._expand_hebbian(primary, include_hebbian, namespace)
        returned = primary + hebbian
        await self._reinforce(returned)
        return returned

    async def _generate_candidates(
        self,
        namespace: str,
        query: str,
        query_embedding: list[float],
        *,
        include_shared: bool,
    ) -> list[Candidate]:
        """Collect recall candidates from complementary retrieval backends.

        RRF is applied here so downstream reranking receives one deduplicated
        candidate list rather than having to know about vector and text sources.

        Args:
            namespace: Memory partition to query.
            query: Lexical query used by trigram search.
            query_embedding: Embedded query used by vector search.
            include_shared: Whether shared/global cross-namespace candidates are
                eligible.

        Returns:
            Deduplicated candidates with normalized RRF scores.
        """
        vector_results = await self.memory_store.vector_candidates(
            namespace,
            query_embedding,
            settings.recall_vector_limit,
            include_shared=include_shared,
        )
        text_results = await self.memory_store.full_text_search_candidates(
            namespace,
            query,
            settings.recall_full_text_search_limit,
            include_shared=include_shared,
        )
        candidates = reciprocal_rank_fusion(
            vector_results,
            text_results,
            rrf_k=settings.rrf_k,
            limit=settings.recall_candidate_limit,
        )
        normalized_rrf = normalize_scores([candidate.rrf_score or 0.0 for candidate in candidates])
        for candidate, score in zip(candidates, normalized_rrf, strict=True):
            candidate.rrf_score = score
        return candidates

    async def _rerank_candidates(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Apply provider relevance when configured without making recall fragile.

        The caller should get a recall response even when reranking is disabled
        or a provider later falls back internally.

        Args:
            query: Recall query passed to the reranker.
            candidates: Fused direct-retrieval candidates.

        Returns:
            The same candidates with optional normalized rerank scores.
        """
        if not settings.rerank_enabled:
            for candidate in candidates:
                candidate.rerank_score = None
            return candidates

        documents = [candidate.memory.raw_content for candidate in candidates]
        rerank_scores = await self.llm_service.rerank_results(query, documents)
        rerank_scores = normalize_scores(rerank_scores)

        for candidate, score in zip(candidates, rerank_scores, strict=False):
            candidate.rerank_score = score
        return candidates

    def _apply_relevance_threshold(self, candidates: list[Candidate]) -> list[Candidate]:
        """Drop weak reranked results only when a reranker judged relevance.

        RRF-only recall is intentionally not thresholded because its scores are
        relative to the candidate set and can be over-aggressive on sparse data.

        Args:
            candidates: Candidates with retrieval or rerank scores.

        Returns:
            Candidates that remain eligible for activation scoring.
        """
        if not settings.rerank_enabled:
            return candidates
        return [
            candidate
            for candidate in candidates
            if candidate.retrieval_score >= settings.rerank_min_score
        ]

    def _score_activation_and_final(
        self,
        candidates: list[Candidate],
        namespace: str,
    ) -> list[Candidate]:
        """Blend relevance, activation, and namespace/scope affinity.

        Scope affinity keeps same-namespace memories preferred while allowing
        relevant shared/global memories to cross namespace boundaries.

        Args:
            candidates: Candidates that passed relevance filtering.
            namespace: Query namespace used to compute scope affinity.

        Returns:
            Candidates sorted by final weighted score.
        """
        for candidate in candidates:
            activation = calculate_activation(
                candidate.memory.base_importance,
                candidate.memory.access_count,
                candidate.memory.last_accessed_at,
                decay_rate=settings.decay_rate,
            )
            affinity = scope_affinity(
                candidate.memory.namespace,
                candidate.memory.scope,
                namespace,
            )
            candidate.activation_score = activation
            candidate.scope_affinity_score = affinity
            activation_for_score = min(activation / 1.5, 1.0)
            candidate.final_score = (
                0.65 * candidate.retrieval_score + 0.25 * activation_for_score + 0.10 * affinity
            )

        candidates.sort(key=lambda candidate: candidate.final_score or 0.0, reverse=True)
        return candidates

    def _select_primary(self, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        """Choose direct answers before any graph expansion is allowed.

        Args:
            candidates: Final-score-sorted candidates.
            top_k: Maximum number of primary memories to keep.

        Returns:
            Primary candidates, optionally filtered for diversity.
        """
        if settings.diversity_enabled:
            return apply_diversity_filter(
                candidates,
                similarity_threshold=settings.diversity_similarity_threshold,
                max_results=top_k,
            )
        return candidates[:top_k]

    async def _expand_hebbian(
        self,
        primary: list[Candidate],
        include_hebbian: bool,
        namespace: str,
    ) -> list[Candidate]:
        """Append strong graph context without reranking it against the query.

        Hebbian spread models associative recall. It is kept separate from the
        primary ranker so graph neighbors cannot push out directly relevant
        memories during Phase 2.

        Args:
            primary: Directly retrieved memories selected for the response.
            include_hebbian: Per-request switch for graph expansion.
            namespace: Query namespace used to keep graph context recallable.

        Returns:
            Graph-linked candidates scored by edge strength and activation.
        """
        if not settings.hebbian_spread_enabled or not include_hebbian or not primary:
            return []

        primary_ids = [candidate.memory.id for candidate in primary]
        neighbor_links = await self.graph_store.get_strong_neighbors(
            primary_ids,
            min_weight=settings.hebbian_edge_threshold,
            limit_per_source=settings.hebbian_spread_limit,
        )
        if not neighbor_links:
            return []

        primary_id_set = set(primary_ids)
        link_by_neighbor: dict[UUID, tuple[UUID, float]] = {}
        for source_id, neighbor_id, weight in neighbor_links:
            if neighbor_id in primary_id_set:
                continue
            current = link_by_neighbor.get(neighbor_id)
            if current is None or weight > current[1]:
                link_by_neighbor[neighbor_id] = (source_id, weight)

        memories = await self.memory_store.get_memories_by_ids(list(link_by_neighbor))
        memory_by_id = {memory.id: memory for memory in memories}
        candidates: list[Candidate] = []

        for neighbor_id, (source_id, edge_weight) in link_by_neighbor.items():
            memory = memory_by_id.get(neighbor_id)
            if memory is None:
                continue
            if memory.namespace != namespace and memory.scope not in {"shared", "global"}:
                continue
            activation = calculate_activation(
                memory.base_importance,
                memory.access_count,
                memory.last_accessed_at,
                decay_rate=settings.decay_rate,
            )
            edge_score = min(edge_weight / 10.0, 1.0)
            activation_for_score = min(activation / 1.5, 1.0)
            affinity = scope_affinity(memory.namespace, memory.scope, namespace)
            candidates.append(
                Candidate(
                    memory=memory,
                    activation_score=activation,
                    scope_affinity_score=affinity,
                    final_score=(0.55 * edge_score + 0.35 * activation_for_score + 0.10 * affinity),
                    source="hebbian",
                    linked_from=source_id,
                    edge_weight=edge_weight,
                )
            )

        candidates.sort(key=lambda candidate: candidate.final_score or 0.0, reverse=True)
        return candidates[: settings.hebbian_spread_limit]

    async def _reinforce(self, candidates: list[Candidate]) -> None:
        """Strengthen returned memories after recall so the graph stays plastic.

        Args:
            candidates: Every memory surfaced to the caller, including Hebbian
                additions.
        """
        if not candidates:
            return

        memory_ids = [candidate.memory.id for candidate in candidates]
        if settings.reinforcement_enabled:
            await self.memory_store.mark_accessed(memory_ids)
            await self.graph_store.reinforce_clique(
                memory_ids,
                increment=settings.reinforcement_edge_increment,
            )
