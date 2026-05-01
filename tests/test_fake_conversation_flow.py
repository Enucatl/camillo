import pytest

from camillo.cognitive.ingestion_service import IngestionService
from camillo.cognitive.recall_service import RecallService
from tests.fakes import FakeGraphStore, FakeLLMService, FakeMemoryStore

FAKE_CONVERSATIONS = [
    {
        "namespace": "repo:camillo",
        "session_id": "architecture-session",
        "turns": [
            (
                "We need this memory service to stay Postgres-native.",
                "I will keep pgvector and trigram search as the storage substrate.",
            ),
            (
                "The main API container should be called teatro.",
                "I will use teatro as the FastAPI service name in Compose.",
            ),
            (
                "The migration runner name db_migrator is boring.",
                "I will call the one-shot Alembic runner sipario.",
            ),
        ],
    },
    {
        "namespace": "repo:camillo",
        "session_id": "observability-session",
        "turns": [
            (
                "I have Phoenix running at phoenix.docker.home.arpa.",
                "I will enable optional Phoenix tracing for LiteLLM calls.",
            ),
            (
                "For development, use OpenRouter models.",
                "I will configure bge-m3 embeddings, Cohere rerank, and Gemma completion.",
            ),
        ],
    },
    {
        "namespace": "repo:billing",
        "session_id": "other-project-session",
        "turns": [
            (
                "This billing project uses Redis for transient invoice locks.",
                "I will keep that Redis detail separate from Camillo memories.",
            ),
        ],
    },
]


@pytest.mark.asyncio
async def test_fake_conversations_flow_through_ingest_recall_and_reinforcement() -> None:
    """Protect the ingest-to-recall path across namespaces and session edges."""
    memory_store = FakeMemoryStore()
    graph_store = FakeGraphStore()
    llm_service = FakeLLMService(valence=0.82)
    ingestion = IngestionService(memory_store, graph_store, llm_service)
    recall = RecallService(memory_store, graph_store, llm_service)

    for conversation in FAKE_CONVERSATIONS:
        for user_msg, ai_msg in conversation["turns"]:
            await ingestion.ingest_interaction(
                namespace=conversation["namespace"],
                user_msg=user_msg,
                ai_msg=ai_msg,
                session_id=conversation["session_id"],
            )

    assert len(memory_store.memories) == 6

    architecture_edges = [edge for edge in graph_store.edges.values() if edge == 1.0]
    assert len(architecture_edges) == 3

    teatro_results = await recall.recall(
        namespace="repo:camillo",
        query="What is the FastAPI container called?",
        top_k=3,
    )

    assert any("teatro" in result.memory.raw_content for result in teatro_results)
    assert all(result.memory.namespace == "repo:camillo" for result in teatro_results)
    assert all("billing project" not in result.memory.raw_content for result in teatro_results)
    assert all(result.memory.id in memory_store.marked_accessed for result in teatro_results)

    phoenix_results = await recall.recall(
        namespace="repo:camillo",
        query="Where should LiteLLM traces go?",
        top_k=2,
    )

    assert any("Phoenix" in result.memory.raw_content for result in phoenix_results)
    assert any(memory.access_count > 0 for memory in memory_store.memories)
