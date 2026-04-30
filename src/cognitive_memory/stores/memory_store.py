from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognitive_memory.db.models import Memory


class MemoryStore:
    def __init__(self, db: AsyncSession):
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
        distance = Memory.embedding.cosine_distance(embedding).label("distance")
        result = await self.db.execute(
            select(Memory, distance)
            .where(Memory.namespace == namespace, Memory.status == "active")
            .order_by(text("distance"))
            .limit(limit)
        )
        return [(memory, 1.0 - float(raw_distance)) for memory, raw_distance in result.all()]

    async def fts_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        similarity = func.similarity(Memory.raw_content, query).label("similarity")
        result = await self.db.execute(
            select(Memory, similarity)
            .where(Memory.namespace == namespace, Memory.status == "active")
            .order_by(desc(text("similarity")))
            .limit(limit)
        )
        return [(memory, float(score)) for memory, score in result.all()]

    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
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
