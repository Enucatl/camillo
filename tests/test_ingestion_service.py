import pytest

from camillo.cognitive.ingestion_service import IngestionService, score_interaction_importance
from tests.fakes import FakeGraphStore, FakeLLMService, FakeMemoryStore


@pytest.mark.asyncio
async def test_ingest_interaction_stores_raw_episodic_memory() -> None:
    """Keep ingestion storing the full conversation turn as episodic memory."""
    memory_store = FakeMemoryStore()
    graph_store = FakeGraphStore()
    llm_service = FakeLLMService()
    service = IngestionService(memory_store, graph_store, llm_service)

    memory = await service.ingest_interaction(
        namespace="repo:backend",
        user_msg="Use Postgres with pgvector.",
        ai_msg="I will remember the Postgres-native choice.",
        session_id="session-1",
    )

    assert memory.namespace == "repo:backend"
    assert memory.session_id == "session-1"
    assert memory.type == "episodic"
    assert memory.base_importance == 0.75
    assert "User:\nUse Postgres with pgvector." in memory.raw_content
    assert "Assistant:\nI will remember" in memory.raw_content
    assert len(memory.embedding) == 32
    assert memory_store.memories[0].raw_content.startswith("User:\nUse Postgres with pgvector.")


def test_score_interaction_importance_is_rule_based() -> None:
    """Keep ingestion importance deterministic and provider-free."""
    assert score_interaction_importance("hello", "hi") == 0.45
    assert (
        score_interaction_importance(
            "Remember: always run pytest before Docker changes.",
            "I will keep that constraint in mind.",
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_ingest_interaction_links_adjacent_session_memories() -> None:
    """Link consecutive turns from the same session in the memory graph."""
    memory_store = FakeMemoryStore()
    graph_store = FakeGraphStore()
    service = IngestionService(memory_store, graph_store, FakeLLMService())

    first = await service.ingest_interaction("repo:backend", "first", "stored", "session-1")
    second = await service.ingest_interaction("repo:backend", "second", "stored", "session-1")

    edge = tuple(sorted((first.id, second.id), key=str))
    assert graph_store.edges[edge] == 1.0


@pytest.mark.asyncio
async def test_ingest_interaction_does_not_link_different_sessions() -> None:
    """Avoid cross-session graph edges when sessions do not match."""
    memory_store = FakeMemoryStore()
    graph_store = FakeGraphStore()
    service = IngestionService(memory_store, graph_store, FakeLLMService())

    await service.ingest_interaction("repo:backend", "first", "stored", "session-1")
    await service.ingest_interaction("repo:backend", "second", "stored", "session-2")

    assert graph_store.edges == {}
