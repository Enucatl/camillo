from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.cognitive.cognitive_math import calculate_activation
from camillo.cognitive.scope_policy import normalize_memory_scope
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
        *,
        confidence: float | None = None,
        source: str | None = None,
        status: str = "active",
        scope: str | None = None,
    ) -> Memory:
        """Insert a memory without committing the surrounding transaction."""
        normalized_scope = normalize_memory_scope(scope, memory_type)
        memory = Memory(
            namespace=namespace,
            raw_content=raw_content,
            embedding=embedding,
            type=memory_type,
            scope=normalized_scope,
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
        *,
        include_shared: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Use vector recall for semantic candidates within one namespace.

        Args:
            namespace: Partition that prevents cross-project recall.
            embedding: Query embedding in the configured vector dimension.
            limit: Maximum semantic candidates to return.
            include_shared: Whether shared/global memories from other namespaces
                are eligible.

        Returns:
            Active memories with cosine-distance-derived similarity scores.
        """
        namespace_filter = (
            (Memory.namespace == namespace) | (Memory.scope.in_(["shared", "global"]))
            if include_shared
            else Memory.namespace == namespace
        )
        distance = Memory.embedding.cosine_distance(embedding).label("distance")
        result = await self.db.execute(
            select(Memory, distance)
            .where(
                namespace_filter,
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
        *,
        include_shared: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Use lexical recall to recover exact terms vector search may miss.

        Args:
            namespace: Partition that prevents cross-project recall.
            query: Raw query text for trigram similarity.
            limit: Maximum lexical candidates to return.
            include_shared: Whether shared/global memories from other namespaces
                are eligible.

        Returns:
            Active memories with positive trigram similarity scores.
        """
        namespace_filter = (
            (Memory.namespace == namespace) | (Memory.scope.in_(["shared", "global"]))
            if include_shared
            else Memory.namespace == namespace
        )
        similarity = func.similarity(Memory.raw_content, query).label("similarity")
        result = await self.db.execute(
            select(Memory, similarity)
            .where(
                namespace_filter,
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
        *,
        include_shared: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Keep the shorter Phase 2 name available without duplicating logic.

        Args:
            namespace: Partition that prevents cross-project recall.
            query: Raw query text for lexical matching.
            limit: Maximum lexical candidates to return.
            include_shared: Whether shared/global memories from other namespaces
                are eligible.

        Returns:
            The same candidates as `full_text_search_candidates`.
        """
        return await self.full_text_search_candidates(
            namespace,
            query,
            limit,
            include_shared=include_shared,
        )

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

    async def select_dream_seeds(
        self,
        namespace: str,
        *,
        limit: int,
        min_activation: float,
        decay_rate: float,
        max_age_days: int | None = None,
    ) -> list[Memory]:
        """Select active episodic memories eligible to initiate dreaming.

        Consolidated episodic rows remain available for audit and optional
        recall, but this query is the hard anti-repeat boundary for dreaming.

        Args:
            namespace: Partition whose episodic memories should be considered.
            limit: Maximum number of seed memories to return.
            min_activation: Minimum ACT-R activation required for a seed.
            decay_rate: Recency decay rate passed to activation scoring.
            max_age_days: Optional age cap for seed candidates.

        Returns:
            Active episodic memories sorted by descending activation.
        """
        if limit <= 0:
            return []

        filters = [
            Memory.namespace == namespace,
            Memory.type == "episodic",
            Memory.status == "active",
        ]
        if max_age_days is not None:
            filters.append(Memory.created_at >= datetime.now(UTC) - timedelta(days=max_age_days))

        result = await self.db.execute(select(Memory).where(*filters))
        scored = []
        for memory in result.scalars().all():
            activation = calculate_activation(
                memory.base_importance,
                memory.access_count,
                memory.last_accessed_at,
                decay_rate=decay_rate,
            )
            if activation >= min_activation:
                scored.append((memory, activation))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory for memory, _activation in scored[:limit]]

    async def get_active_episodic_by_ids(
        self,
        memory_ids: list[UUID],
        *,
        namespace: str,
    ) -> list[Memory]:
        """Fetch dreamable memories from a graph traversal.

        Args:
            memory_ids: IDs discovered through Hebbian traversal.
            namespace: Partition guard for multi-tenant memory storage.

        Returns:
            Active episodic memories matching the supplied IDs.
        """
        if not memory_ids:
            return []

        result = await self.db.execute(
            select(Memory).where(
                Memory.id.in_(memory_ids),
                Memory.namespace == namespace,
                Memory.type == "episodic",
                Memory.status == "active",
            )
        )
        return list(result.scalars().all())

    async def mark_memories_consolidated_after_dream(
        self,
        memory_ids: list[UUID],
        *,
        created_memory_ids: list[UUID],
        penalty: float,
        min_importance: float,
        dream_run_id: UUID,
    ) -> None:
        """Mark source episodes as consolidated after successful promotion.

        Args:
            memory_ids: Source episodic memories that supported the dream.
            created_memory_ids: Semantic memories created or reinforced.
            penalty: Fractional importance reduction for source episodes.
            min_importance: Lower bound for source episode importance.
            dream_run_id: Audit run that performed consolidation.
        """
        if not memory_ids:
            return

        result = await self.db.execute(
            select(Memory).where(
                Memory.id.in_(memory_ids),
                Memory.type == "episodic",
                Memory.status == "active",
            )
        )
        now = datetime.now(UTC).isoformat()
        consolidated_into = [str(memory_id) for memory_id in created_memory_ids]
        for memory in result.scalars().all():
            metadata = dict(memory.metadata_json or {})
            metadata["dreaming"] = {
                **dict(metadata.get("dreaming") or {}),
                "consolidated_at": now,
                "consolidated_into": consolidated_into,
                "dream_run_id": str(dream_run_id),
            }
            memory.status = "consolidated"
            memory.base_importance = max(memory.base_importance * (1.0 - penalty), min_importance)
            memory.metadata_json = metadata
        await self.db.flush()

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

    async def update_memory_status(
        self,
        memory_id: UUID,
        status: str,
        *,
        reason: str | None = None,
        superseded_by: UUID | None = None,
    ) -> Memory | None:
        """Apply lifecycle changes while reducing inactive memory strength.

        Args:
            memory_id: Memory row to update.
            status: Lifecycle status chosen by reconciliation.
            reason: Optional human-readable reason for auditability.
            superseded_by: Replacement memory when `status` is superseded.

        Returns:
            The updated memory when it exists, otherwise `None`.
        """
        memory = await self.db.get(Memory, memory_id)
        if memory is None:
            return None

        memory.status = status
        memory.status_reason = reason
        if status in {"deprecated", "superseded"}:
            memory.deprecated_at = datetime.now(UTC)
            memory.base_importance = min(memory.base_importance, 0.2)
        if superseded_by is not None:
            memory.superseded_by = superseded_by
        await self.db.flush()
        return memory

    async def reinforce_memory(
        self,
        memory_id: UUID,
        *,
        increment_access: bool = True,
        importance_boost: float = 0.05,
    ) -> Memory | None:
        """Strengthen a known memory instead of creating a duplicate.

        Args:
            memory_id: Existing memory to reinforce.
            increment_access: Whether to increment recall-like access count.
            importance_boost: Bounded increase in base importance.

        Returns:
            The updated memory when it exists, otherwise `None`.
        """
        memory = await self.db.get(Memory, memory_id)
        if memory is None:
            return None

        if increment_access:
            memory.access_count += 1
        memory.base_importance = min(memory.base_importance + importance_boost, 1.0)
        memory.last_accessed_at = datetime.now(UTC)
        await self.db.flush()
        return memory

    async def memory_stats(self, namespace: str) -> dict[str, Any]:
        """Return operational memory counts for one namespace.

        Args:
            namespace: Partition whose memories should be counted.

        Returns:
            Total, type counts, and status counts in a JSON-serializable dict.
        """
        total = await self.db.scalar(
            select(func.count()).select_from(Memory).where(Memory.namespace == namespace)
        )
        by_type_rows = await self.db.execute(
            select(Memory.type, func.count())
            .where(Memory.namespace == namespace)
            .group_by(Memory.type)
        )
        by_status_rows = await self.db.execute(
            select(Memory.status, func.count())
            .where(Memory.namespace == namespace)
            .group_by(Memory.status)
        )
        by_scope_rows = await self.db.execute(
            select(Memory.scope, func.count())
            .where(Memory.namespace == namespace)
            .group_by(Memory.scope)
        )
        return {
            "namespace": namespace,
            "total": int(total or 0),
            "by_type": {memory_type: count for memory_type, count in by_type_rows.all()},
            "by_status": {status: count for status, count in by_status_rows.all()},
            "by_scope": {scope: count for scope, count in by_scope_rows.all()},
        }
