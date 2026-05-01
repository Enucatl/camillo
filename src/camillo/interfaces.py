from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from camillo.db.models import Memory


class EmbeddingProvider(ABC):
    """Abstract adapter for text embedding providers."""

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Embed text into the vector space used by the memory store."""


class CompletionProvider(ABC):
    """Abstract adapter for LLM completion behavior used by cognition services."""

    @abstractmethod
    async def score_valence(self, raw_content: str) -> float:
        """Score long-term memory importance on a continuous 0.0-1.0 scale."""


class Reranker(ABC):
    """Separate reranking from retrieval so providers remain swappable."""

    @abstractmethod
    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Allow recall to ask for relevance without knowing provider details.

        Args:
            query: User recall query.
            documents: Candidate texts in original candidate order.

        Returns:
            One relevance score per document in the same order.
        """


class MemoryStoreProtocol(ABC):
    """Abstract persistence boundary for cognitive memories."""

    @abstractmethod
    async def insert_memory(
        self,
        namespace: str,
        raw_content: str,
        embedding: list[float],
        memory_type: str,
        base_importance: float,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Persist a memory and return the database-backed model."""

    @abstractmethod
    async def get_previous_memory_in_session(
        self, namespace: str, session_id: str
    ) -> Memory | None:
        """Find the latest active memory in the same conversation session."""

    @abstractmethod
    async def vector_candidates(
        self,
        namespace: str,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Return vector-similar memories with normalized similarity scores."""

    @abstractmethod
    async def full_text_search_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Return lexical full-text-search candidates with relevance scores."""

    @abstractmethod
    async def get_memories_by_ids(
        self,
        memory_ids: list[UUID],
        *,
        active_only: bool = True,
    ) -> list[Memory]:
        """Let graph expansion hydrate neighbors without exposing inactive rows.

        Args:
            memory_ids: IDs discovered outside direct retrieval.
            active_only: Whether hidden/inactive memories should be excluded.

        Returns:
            Matching memory models.
        """

    @abstractmethod
    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        """Record that the recall path surfaced the selected memories."""


class GraphStoreProtocol(ABC):
    """Abstract persistence boundary for Hebbian memory edges."""

    @abstractmethod
    async def create_or_increment_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        increment: float = 1.0,
    ) -> None:
        """Create or strengthen an association between two memories."""

    @abstractmethod
    async def reinforce_clique(self, memory_ids: list[UUID], increment: float = 1.0) -> None:
        """Strengthen pairwise associations among co-recalled memories."""

    @abstractmethod
    async def get_strong_neighbors(
        self,
        memory_ids: list[UUID],
        *,
        min_weight: float,
        limit_per_source: int,
    ) -> list[tuple[UUID, UUID, float]]:
        """Expose graph context without coupling recall to edge storage details.

        Args:
            memory_ids: Primary memories anchoring the expansion.
            min_weight: Minimum association strength to return.
            limit_per_source: Per-anchor cap for spreading.

        Returns:
            Source ID, neighbor ID, and edge weight tuples.
        """
