from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from camillo.db.models import Memory


class EmbeddingProvider(ABC):
    """Abstract adapter for the configured embedding provider."""

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Embed text in the memory store's vector space."""


class Reranker(ABC):
    """Optional provider adapter for final relevance refinement."""

    @abstractmethod
    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score for each document."""


class MemoryStoreProtocol(ABC):
    """Persistence boundary for memories and their lifecycle."""

    @abstractmethod
    async def insert_memory(
        self,
        raw_content: str,
        embedding: list[float],
        memory_type: str,
        base_importance: float,
        workspace: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        confidence: float | None = None,
        source: str | None = None,
        status: str = "active",
    ) -> Memory:
        """Persist one memory without committing the caller's transaction."""

    @abstractmethod
    async def get_previous_memory_in_session(self, session_id: str) -> Memory | None:
        """Find the newest active memory in a conversation session."""

    @abstractmethod
    async def vector_candidates(
        self, embedding: list[float], limit: int
    ) -> list[tuple[Memory, float]]:
        """Return active vector candidates from the whole corpus."""

    @abstractmethod
    async def full_text_search_candidates(
        self, query: str, limit: int
    ) -> list[tuple[Memory, float]]:
        """Return active lexical candidates from the whole corpus."""

    @abstractmethod
    async def get_memories_by_ids(
        self, memory_ids: list[UUID], *, active_only: bool = True
    ) -> list[Memory]:
        """Fetch explicit memory IDs without applying workspace filtering."""

    @abstractmethod
    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        """Record public recall access for selected memories."""

    @abstractmethod
    async def update_memory_status(
        self,
        memory_id: UUID,
        status: str,
        *,
        reason: str | None = None,
        superseded_by: UUID | None = None,
    ) -> Memory | None:
        """Apply one explicit lifecycle transition."""

    @abstractmethod
    async def reinforce_memory(
        self, memory_id: UUID, *, increment_access: bool = True
    ) -> Memory | None:
        """Reinforce one existing memory after duplicate submission."""

    @abstractmethod
    async def memory_stats(self, workspace: str | None = None) -> dict[str, Any]:
        """Return corpus counts, optionally restricted to a workspace hint."""
