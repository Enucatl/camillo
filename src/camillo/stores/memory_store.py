from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.db.models import Memory
from camillo.interfaces import MemoryStoreProtocol


class MemoryStore(MemoryStoreProtocol):
    """Postgres implementation of the memory persistence boundary."""

    def __init__(self, db: AsyncSession):
        """Initialize the store with a request-scoped async session.

        Args:
            db: Async SQLAlchemy session owned by the caller.
        """
        self.db = db

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
        """Insert a memory without committing the surrounding transaction."""
        memory = Memory(
            namespace=namespace,
            raw_content=raw_content,
            embedding=embedding,
            type=memory_type,
            base_importance=base_importance,
            session_id=session_id,
            metadata_json=metadata or {},
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def get_previous_memory_in_session(
        self, namespace: str, session_id: str
    ) -> Memory | None:
        """Fetch the newest active memory in a session to support adjacency links."""
        result = await self.db.execute(
            select(Memory)
            .where(
                Memory.namespace == namespace,
                Memory.session_id == session_id,
                Memory.status == "active",
            )
            .order_by(desc(Memory.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def vector_candidates(
        self,
        namespace: str,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Use vector recall for semantic candidates within one namespace.

        Args:
            namespace: Partition that prevents cross-project recall.
            embedding: Query embedding in the configured vector dimension.
            limit: Maximum semantic candidates to return.

        Returns:
            Active memories with cosine-distance-derived similarity scores.
        """
        distance = Memory.embedding.cosine_distance(embedding).label("distance")
        result = await self.db.execute(
            select(Memory, distance)
            .where(
                Memory.namespace == namespace,
                Memory.status == "active",
                Memory.embedding.is_not(None),
            )
            .order_by(text("distance"))
            .limit(limit)
        )
        return [(memory, 1.0 - float(raw_distance)) for memory, raw_distance in result.all()]

    async def full_text_search_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Use lexical recall to recover exact terms vector search may miss.

        Args:
            namespace: Partition that prevents cross-project recall.
            query: Raw query text for trigram similarity.
            limit: Maximum lexical candidates to return.

        Returns:
            Active memories with positive trigram similarity scores.
        """
        similarity = func.similarity(Memory.raw_content, query).label("similarity")
        result = await self.db.execute(
            select(Memory, similarity)
            .where(
                Memory.namespace == namespace,
                Memory.status == "active",
                similarity > 0.0,
            )
            .order_by(desc(text("similarity")))
            .limit(limit)
        )
        return [(memory, float(score)) for memory, score in result.all()]

    async def fts_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Keep the shorter Phase 2 name available without duplicating logic.

        Args:
            namespace: Partition that prevents cross-project recall.
            query: Raw query text for lexical matching.
            limit: Maximum lexical candidates to return.

        Returns:
            The same candidates as `full_text_search_candidates`.
        """
        return await self.full_text_search_candidates(namespace, query, limit)

    async def get_memories_by_ids(
        self,
        memory_ids: list[UUID],
        *,
        active_only: bool = True,
    ) -> list[Memory]:
        """Support graph expansion while preserving recall visibility rules.

        Args:
            memory_ids: IDs discovered from Hebbian edges.
            active_only: Whether inactive memories should stay hidden.

        Returns:
            Matching memories, unordered by design because callers own ranking.
        """
        if not memory_ids:
            return []

        filters = [Memory.id.in_(memory_ids)]
        if active_only:
            filters.append(Memory.status == "active")

        result = await self.db.execute(select(Memory).where(*filters))
        return list(result.scalars().all())

    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        """Update recall bookkeeping for memories surfaced to a caller."""
        if not memory_ids:
            return
        await self.db.execute(
            update(Memory)
            .where(Memory.id.in_(memory_ids))
            .values(
                access_count=Memory.access_count + 1,
                last_accessed_at=datetime.now(UTC),
            )
        )
