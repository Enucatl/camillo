from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from cognitive_memory.db.models import Memory


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def get_embedding(self, text: str) -> list[float]: ...


@runtime_checkable
class CompletionProvider(Protocol):
    async def score_valence(self, raw_content: str) -> float: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank_results(self, query: str, documents: list[str]) -> list[float]: ...


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    async def insert_memory(
        self,
        namespace: str,
        raw_content: str,
        embedding: list[float],
        memory_type: str,
        base_importance: float,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory: ...

    async def get_previous_memory_in_session(
        self, namespace: str, session_id: str
    ) -> Memory | None: ...

    async def vector_candidates(
        self,
        namespace: str,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]: ...

    async def fts_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]: ...

    async def mark_accessed(self, memory_ids: list[UUID]) -> None: ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    async def create_or_increment_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        increment: float = 1.0,
    ) -> None: ...

    async def reinforce_clique(self, memory_ids: list[UUID], increment: float = 1.0) -> None: ...
