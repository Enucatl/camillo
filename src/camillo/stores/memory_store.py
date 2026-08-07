from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.cognitive.cognitive_math import calculate_activation
from camillo.db.models import Memory
from camillo.interfaces import MemoryStoreProtocol


class MemoryStore(MemoryStoreProtocol):
    """Persist and retrieve the single user's complete memory corpus."""

    def __init__(self, db: AsyncSession):
        """Bind the store to one request or worker transaction."""
        self.db = db

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
        """Insert a memory while leaving transaction ownership to the caller."""
        memory = Memory(
            workspace=workspace,
            raw_content=raw_content,
            embedding=embedding,
            type=memory_type,
            base_importance=base_importance,
            session_id=session_id,
            metadata_json=metadata or {},
            confidence=confidence if confidence is not None else 0.8,
            source=source,
            status=status,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def get_previous_memory_in_session(self, session_id: str) -> Memory | None:
        """Return the latest active episode in a conversation session."""
        result = await self.db.execute(
            select(Memory)
            .where(Memory.session_id == session_id, Memory.status == "active")
            .order_by(desc(Memory.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def vector_candidates(
        self, embedding: list[float], limit: int
    ) -> list[tuple[Memory, float]]:
        """Find active semantic candidates corpus-wide."""
        distance = Memory.embedding.cosine_distance(embedding).label("distance")
        result = await self.db.execute(
            select(Memory, distance)
            .where(Memory.status == "active", Memory.embedding.is_not(None))
            .order_by(text("distance"))
            .limit(limit)
        )
        return [(memory, 1.0 - float(raw_distance)) for memory, raw_distance in result.all()]

    async def full_text_search_candidates(
        self, query: str, limit: int
    ) -> list[tuple[Memory, float]]:
        """Find active lexical candidates corpus-wide."""
        similarity = func.similarity(Memory.raw_content, query).label("similarity")
        result = await self.db.execute(
            select(Memory, similarity)
            .where(Memory.status == "active", similarity > 0.0)
            .order_by(desc(text("similarity")))
            .limit(limit)
        )
        return [(memory, float(score)) for memory, score in result.all()]

    async def get_memories_by_ids(
        self, memory_ids: list[UUID], *, active_only: bool = True
    ) -> list[Memory]:
        """Fetch explicit IDs for lifecycle and deduplication operations."""
        if not memory_ids:
            return []
        filters = [Memory.id.in_(memory_ids)]
        if active_only:
            filters.append(Memory.status == "active")
        result = await self.db.execute(select(Memory).where(*filters))
        return list(result.scalars().all())

    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        """Update access bookkeeping for public recall results."""
        if memory_ids:
            await self.db.execute(
                update(Memory)
                .where(Memory.id.in_(memory_ids))
                .values(access_count=Memory.access_count + 1, last_accessed_at=datetime.now(UTC))
            )

    async def reinforce_memory(
        self, memory_id: UUID, *, increment_access: bool = True
    ) -> Memory | None:
        """Reinforce one exact duplicate without creating another row."""
        memory = await self.db.get(Memory, memory_id)
        if memory is None or memory.status != "active":
            return None
        if increment_access:
            memory.access_count += 1
        memory.base_importance = min(memory.base_importance + 0.05, 1.0)
        memory.last_accessed_at = datetime.now(UTC)
        await self.db.flush()
        return memory

    async def update_memory_status(
        self,
        memory_id: UUID,
        status: str,
        *,
        reason: str | None = None,
        superseded_by: UUID | None = None,
    ) -> Memory | None:
        """Change lifecycle state for one explicit memory ID."""
        memory = await self.db.get(Memory, memory_id)
        if memory is None or memory.status != "active":
            return None
        memory.status = status
        memory.status_reason = reason
        memory.deprecated_at = datetime.now(UTC) if status in {"deprecated", "superseded"} else None
        if superseded_by is not None:
            memory.superseded_by = superseded_by
        await self.db.flush()
        return memory

    async def memory_stats(self, workspace: str | None = None) -> dict[str, Any]:
        """Count memories without making workspace a retrieval boundary."""
        condition = Memory.workspace == workspace if workspace is not None else True
        total = await self.db.scalar(select(func.count()).select_from(Memory).where(condition))
        by_type = await self.db.execute(
            select(Memory.type, func.count()).where(condition).group_by(Memory.type)
        )
        by_status = await self.db.execute(
            select(Memory.status, func.count()).where(condition).group_by(Memory.status)
        )
        return {
            "total": int(total or 0),
            "by_type": dict(by_type.all()),
            "by_status": dict(by_status.all()),
        }

    async def select_dream_seeds(
        self, *, limit: int, min_activation: float, decay_rate: float
    ) -> list[Memory]:
        """Select unconsolidated episode seeds across the corpus."""
        result = await self.db.execute(
            select(Memory).where(Memory.type == "episode", Memory.status == "active")
        )
        scored = [
            (
                memory,
                calculate_activation(
                    memory.base_importance,
                    memory.access_count,
                    memory.last_accessed_at,
                    decay_rate=decay_rate,
                ),
            )
            for memory in result.scalars().all()
        ]
        scored = [(memory, score) for memory, score in scored if score >= min_activation]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory for memory, _score in scored[:limit]]

    async def mark_memories_consolidated_after_dream(
        self, memory_ids: list[UUID], *, created_memory_ids: list[UUID], dream_run_id: UUID
    ) -> None:
        """Mark source episodes only after promotion or reinforcement succeeds."""
        if not memory_ids:
            return
        result = await self.db.execute(
            select(Memory).where(
                Memory.id.in_(memory_ids), Memory.type == "episode", Memory.status == "active"
            )
        )
        now = datetime.now(UTC).isoformat()
        for memory in result.scalars().all():
            metadata = dict(memory.metadata_json or {})
            metadata["dreaming"] = {
                "consolidated_at": now,
                "created_memory_ids": [str(i) for i in created_memory_ids],
                "dream_run_id": str(dream_run_id),
            }
            memory.metadata_json = metadata
            memory.status = "consolidated"
        await self.db.flush()
