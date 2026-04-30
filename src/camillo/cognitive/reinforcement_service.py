from uuid import UUID

from camillo.interfaces import GraphStoreProtocol, MemoryStoreProtocol


class ReinforcementService:
    """Applies recall reinforcement side effects as one service boundary."""

    def __init__(self, memory_store: MemoryStoreProtocol, graph_store: GraphStoreProtocol):
        """Initialize reinforcement with memory and graph stores."""
        self.memory_store = memory_store
        self.graph_store = graph_store

    async def reinforce_access(self, memory_ids: list[UUID]) -> None:
        """Mark memories as accessed and strengthen their mutual associations."""
        await self.memory_store.mark_accessed(memory_ids)
        await self.graph_store.reinforce_clique(memory_ids)
