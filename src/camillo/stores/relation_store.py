from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.db.models import MemoryRelation


class RelationStore:
    """Persist semantic/lifecycle relations separately from Hebbian edges."""

    def __init__(self, db: AsyncSession):
        """Initialize the store with the caller-owned async session.

        Args:
            db: Request or tool-scoped database session.
        """
        self.db = db

    async def create_relation(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        confidence: float = 0.8,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRelation:
        """Create or refresh a semantic relation without committing.

        Args:
            source_id: Newer/source memory in the relation.
            target_id: Existing/target memory in the relation.
            relation_type: Semantic relation label.
            confidence: Confidence in the relation.
            rationale: Optional classifier rationale.
            metadata: Optional structured reconciliation details.

        Returns:
            The inserted or updated relation row.
        """
        stmt = (
            insert(MemoryRelation)
            .values(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
                rationale=rationale,
                metadata_json=metadata or {},
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_update(
                constraint="uq_memory_relations_source_target_type",
                set_={
                    "confidence": confidence,
                    "rationale": rationale,
                    "metadata_json": metadata or {},
                    "created_at": datetime.now(UTC),
                },
            )
            .returning(MemoryRelation)
        )
        relation = (await self.db.execute(stmt)).scalar_one()
        await self.db.flush()
        return relation

    async def get_relations_for_memory(self, memory_id: UUID) -> list[MemoryRelation]:
        """Fetch relations where a memory is source or target.

        Args:
            memory_id: Memory id whose semantic edges should be loaded.

        Returns:
            Matching relation rows.
        """
        result = await self.db.execute(
            select(MemoryRelation).where(
                or_(
                    MemoryRelation.source_id == memory_id,
                    MemoryRelation.target_id == memory_id,
                )
            )
        )
        return list(result.scalars().all())
