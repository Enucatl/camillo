from uuid import UUID

from camillo.cognitive.cognitive_math import calculate_activation
from camillo.interfaces import EmbeddingProvider, GraphStoreProtocol, MemoryStoreProtocol
from camillo.settings import settings


class RecallService:
    """Coordinates hybrid memory recall and reinforcement."""

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        graph_store: GraphStoreProtocol,
        llm_service: EmbeddingProvider,
    ):
        """Initialize recall with storage and embedding dependencies.

        Args:
            memory_store: Memory persistence port.
            graph_store: Association graph persistence port.
            llm_service: Provider used to embed recall queries.
        """
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.llm_service = llm_service

    async def recall(
        self,
        namespace: str,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """Recall the strongest memories for a namespace-scoped query.

        Args:
            namespace: Logical partition for the caller's memories.
            query: Natural-language recall query.
            top_k: Maximum number of memories to return.

        Returns:
            Serialized memory records with blended retrieval scores.
        """
        query_embedding = await self.llm_service.get_embedding(query)
        vector_candidates = await self.memory_store.vector_candidates(
            namespace,
            query_embedding,
            settings.recall_vector_limit,
        )
        full_text_search_candidates = await self.memory_store.full_text_search_candidates(
            namespace,
            query,
            settings.recall_full_text_search_limit,
        )

        merged: dict[UUID, dict] = {}
        for memory, retrieval_score in [*vector_candidates, *full_text_search_candidates]:
            current = merged.get(memory.id)
            if current is None or retrieval_score > current["retrieval_score"]:
                merged[memory.id] = {"memory": memory, "retrieval_score": retrieval_score}

        scored = []
        for item in merged.values():
            memory = item["memory"]
            retrieval_score = max(0.0, min(float(item["retrieval_score"]), 1.0))
            activation = calculate_activation(
                memory.base_importance,
                memory.access_count,
                memory.last_accessed_at,
                decay_rate=settings.decay_rate,
            )
            score = (0.7 * retrieval_score) + (0.3 * activation)
            scored.append({"memory": memory, "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = scored[:top_k]
        memory_ids = [item["memory"].id for item in selected]
        await self.memory_store.mark_accessed(memory_ids)
        await self.graph_store.reinforce_clique(memory_ids)

        return [
            {
                "id": item["memory"].id,
                "namespace": item["memory"].namespace,
                "raw_content": item["memory"].raw_content,
                "type": item["memory"].type,
                "base_importance": item["memory"].base_importance,
                "access_count": item["memory"].access_count,
                "score": item["score"],
            }
            for item in selected
        ]
