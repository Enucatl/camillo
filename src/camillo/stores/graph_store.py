from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID

from sqlalchemy import or_, select
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
        """Store associations canonically so recall reinforcement is idempotent.

        Args:
            source_id: One memory in the pair.
            target_id: The other memory in the pair.
            increment: Amount to strengthen the association.
        """
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
        """Strengthen co-recalled memories as one associative clique.

        Args:
            memory_ids: Memories surfaced together by recall.
            increment: Amount to add to every unique pair.
        """
        unique_ids = list(dict.fromkeys(memory_ids))
        if len(unique_ids) < 2:
            return

        for source_id, target_id in combinations(unique_ids, 2):
            await self.create_or_increment_edge(source_id, target_id, increment)

    async def get_strong_neighbors(
        self,
        memory_ids: list[UUID],
        *,
        min_weight: float,
        limit_per_source: int,
    ) -> list[tuple[UUID, UUID, float]]:
        """Expose only strong graph context for Hebbian spreading.

        Args:
            memory_ids: Primary memories anchoring graph expansion.
            min_weight: Minimum edge weight worth surfacing as context.
            limit_per_source: Per-anchor cap that prevents one memory from
                flooding the spread set.

        Returns:
            Tuples of anchor ID, neighbor ID, and edge weight.
        """
        source_ids = list(dict.fromkeys(memory_ids))
        if not source_ids or limit_per_source <= 0:
            return []

        source_set = set(source_ids)
        result = await self.db.execute(
            select(HebbianEdge).where(
                HebbianEdge.weight >= min_weight,
                or_(
                    HebbianEdge.source_id.in_(source_ids),
                    HebbianEdge.target_id.in_(source_ids),
                ),
            )
        )

        by_anchor: dict[UUID, list[tuple[UUID, UUID, float, datetime]]] = {
            memory_id: [] for memory_id in source_ids
        }
        for edge in result.scalars().all():
            if edge.source_id in source_set and edge.target_id not in source_set:
                by_anchor[edge.source_id].append(
                    (edge.source_id, edge.target_id, edge.weight, edge.last_co_accessed_at)
                )
            if edge.target_id in source_set and edge.source_id not in source_set:
                by_anchor[edge.target_id].append(
                    (edge.target_id, edge.source_id, edge.weight, edge.last_co_accessed_at)
                )

        best_by_neighbor: dict[UUID, tuple[UUID, UUID, float, datetime]] = {}
        for links in by_anchor.values():
            links.sort(key=lambda item: (item[2], item[3]), reverse=True)
            for link in links[:limit_per_source]:
                current = best_by_neighbor.get(link[1])
                if current is None or (link[2], link[3]) > (current[2], current[3]):
                    best_by_neighbor[link[1]] = link

        links = list(best_by_neighbor.values())
        links.sort(key=lambda item: (item[2], item[3]), reverse=True)
        return [(source_id, neighbor_id, weight) for source_id, neighbor_id, weight, _ in links]
