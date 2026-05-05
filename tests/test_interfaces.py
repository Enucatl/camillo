from camillo.ai.llm_service import LiteLLMService
from camillo.interfaces import (
    CompletionProvider,
    EmbeddingProvider,
    GraphStoreProtocol,
    MemoryStoreProtocol,
    Reranker,
)
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore


def test_litellm_service_satisfies_provider_interfaces() -> None:
    """Keep the service aligned with the provider protocol surface."""
    service = LiteLLMService()

    assert isinstance(service, CompletionProvider)
    assert isinstance(service, EmbeddingProvider)
    assert isinstance(service, Reranker)


def test_store_implementations_satisfy_protocols() -> None:
    """Ensure the concrete stores continue to satisfy the protocol contracts."""
    memory_store = MemoryStore(db=object())  # type: ignore[arg-type]
    graph_store = GraphStore(db=object())  # type: ignore[arg-type]

    assert isinstance(memory_store, MemoryStoreProtocol)
    assert isinstance(graph_store, GraphStoreProtocol)
