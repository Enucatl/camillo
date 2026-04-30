from camillo.db.models import Memory
from camillo.interfaces import CompletionProvider, EmbeddingProvider, GraphStoreProtocol
from camillo.interfaces import MemoryStoreProtocol


class IngestionService:
    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        graph_store: GraphStoreProtocol,
        llm_service: CompletionProvider | EmbeddingProvider,
    ):
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.llm_service = llm_service

    async def ingest_interaction(
        self,
        namespace: str,
        user_msg: str,
        ai_msg: str,
        session_id: str | None,
    ) -> Memory:
        raw_content = f"User:\n{user_msg}\n\nAssistant:\n{ai_msg}"
        base_importance = await self.llm_service.score_valence(raw_content)
        embedding = await self.llm_service.get_embedding(raw_content)
        previous_memory = None
        if session_id is not None:
            previous_memory = await self.memory_store.get_previous_memory_in_session(
                namespace,
                session_id,
            )

        memory = await self.memory_store.insert_memory(
            namespace=namespace,
            raw_content=raw_content,
            embedding=embedding,
            memory_type="episodic",
            base_importance=base_importance,
            session_id=session_id,
        )

        if previous_memory is not None:
            await self.graph_store.create_or_increment_edge(previous_memory.id, memory.id)

        return memory
