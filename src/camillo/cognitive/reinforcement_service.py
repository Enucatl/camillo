from uuid import UUID

from camillo.interfaces import GraphStoreProtocol, MemoryStoreProtocol


class ReinforcementService:
    def __init__(self, memory_store: MemoryStoreProtocol, graph_store: GraphStoreProtocol):
        self.memory_store = memory_store
        self.graph_store = graph_store

    async def reinforce_access(self, memory_ids: list[UUID]) -> None:
        await self.memory_store.mark_accessed(memory_ids)
        await self.graph_store.reinforce_clique(memory_ids)
