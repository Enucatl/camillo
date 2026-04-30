from cognitive_memory.ai.llm_service import LiteLLMService
from cognitive_memory.interfaces import CompletionProvider, EmbeddingProvider, Reranker
from cognitive_memory.interfaces import GraphStoreProtocol, MemoryStoreProtocol
from cognitive_memory.stores.graph_store import GraphStore
from cognitive_memory.stores.memory_store import MemoryStore


def test_litellm_service_satisfies_provider_interfaces() -> None:
    service = LiteLLMService()

    assert isinstance(service, CompletionProvider)
    assert isinstance(service, EmbeddingProvider)
    assert isinstance(service, Reranker)


def test_store_implementations_satisfy_protocols() -> None:
    memory_store = MemoryStore(db=object())  # type: ignore[arg-type]
    graph_store = GraphStore(db=object())  # type: ignore[arg-type]

    assert isinstance(memory_store, MemoryStoreProtocol)
    assert isinstance(graph_store, GraphStoreProtocol)
