from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from camillo.db.models import DreamRun


class DreamStore:
    """Persist dreaming run bookkeeping without owning transactions."""

    def __init__(self, db: AsyncSession):
        """Initialize the store with a caller-owned async session.

        Args:
            db: Request, worker, or test session that controls commit/rollback.
        """
        self.db = db

    async def create_run(
        self,
        namespace: str,
        *,
        dry_run: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DreamRun:
        """Create a run row before seed selection starts.

        Args:
            namespace: Memory partition being consolidated.
            dry_run: Whether the run is observational only.
            metadata: Optional settings snapshot for auditability.

        Returns:
            The flushed dream run model.
        """
        dream_run = DreamRun(
            namespace=namespace,
            status="dry_run" if dry_run else "running",
            started_at=datetime.now(UTC),
            metadata_json=metadata or {},
        )
        self.db.add(dream_run)
        await self.db.flush()
        return dream_run

    async def complete_run(
        self,
        dream_run_id: UUID,
        *,
        seed_memory_ids: list[UUID],
        source_memory_ids: list[UUID],
        created_memory_ids: list[UUID],
        clusters_considered: int,
        clusters_dreamed: int,
        memories_created: int,
        dry_run: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DreamRun:
        """Mark a run complete after all eligible clusters are processed.

        Args:
            dream_run_id: Run row to update.
            seed_memory_ids: Seeds selected at the start of the run.
            source_memory_ids: Episodic memories included in processed clusters.
            created_memory_ids: Semantic memories created or reinforced.
            clusters_considered: Number of non-duplicate clusters inspected.
            clusters_dreamed: Number of clusters with accepted dream output.
            memories_created: Number of accepted durable memory outcomes.
            dry_run: Whether the run avoided side-effect writes.
            metadata: Optional completion metadata to merge into the row.

        Returns:
            The flushed dream run model.
        """
        dream_run = await self.db.get(DreamRun, dream_run_id)
        if dream_run is None:
            raise ValueError(f"Dream run {dream_run_id} was not found")

        dream_run.status = "dry_run" if dry_run else "completed"
        dream_run.completed_at = datetime.now(UTC)
        dream_run.seed_memory_ids = seed_memory_ids
        dream_run.source_memory_ids = source_memory_ids
        dream_run.created_memory_ids = created_memory_ids
        dream_run.clusters_considered = clusters_considered
        dream_run.clusters_dreamed = clusters_dreamed
        dream_run.memories_created = memories_created
        if metadata:
            dream_run.metadata_json = {**(dream_run.metadata_json or {}), **metadata}
        await self.db.flush()
        return dream_run

    async def fail_run(self, dream_run_id: UUID, error: str) -> DreamRun:
        """Record a failed dreaming run without swallowing the caller error.

        Args:
            dream_run_id: Run row to update.
            error: Human-readable failure detail.

        Returns:
            The flushed dream run model.
        """
        dream_run = await self.db.get(DreamRun, dream_run_id)
        if dream_run is None:
            raise ValueError(f"Dream run {dream_run_id} was not found")

        dream_run.status = "failed"
        dream_run.completed_at = datetime.now(UTC)
        dream_run.error = error
        await self.db.flush()
        return dream_run
