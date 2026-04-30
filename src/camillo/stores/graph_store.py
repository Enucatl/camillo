from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.db.models import HebbianEdge
from camillo.interfaces import GraphStoreProtocol


class GraphStore(GraphStoreProtocol):
    """Postgres implementation of the Hebbian graph persistence boundary."""

    def __init__(self, db: AsyncSession):
        """Initialize the graph store with the caller-owned async session.

        Args:
            db: Async SQLAlchemy session used for graph mutations.
        """
        self.db = db

    async def create_or_increment_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        increment: float = 1.0,
    ) -> None:
        """Create or strengthen an undirected association between two memories."""
        if source_id == target_id:
            return

        # Store undirected edges canonically so pair lookups cannot duplicate.
        ordered_source, ordered_target = sorted((source_id, target_id), key=str)
        result = await self.db.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id == ordered_source,
                HebbianEdge.target_id == ordered_target,
            )
        )
        edge = result.scalar_one_or_none()
        now = datetime.now(UTC)
        if edge is None:
            self.db.add(
                HebbianEdge(
                    source_id=ordered_source,
                    target_id=ordered_target,
                    weight=increment,
                    last_co_accessed_at=now,
                )
            )
            await self.db.flush()
            return

        edge.weight += increment
        edge.last_co_accessed_at = now
        await self.db.flush()

    async def reinforce_clique(self, memory_ids: list[UUID], increment: float = 1.0) -> None:
        """Reinforce every unique memory pair from a recall result set."""
        for source_id, target_id in combinations(dict.fromkeys(memory_ids), 2):
            await self.create_or_increment_edge(source_id, target_id, increment)
