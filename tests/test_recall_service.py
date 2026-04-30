import random
import time

import pytest

from camillo.cognitive.recall_service import RecallService
from tests.fakes import FakeGraphStore, FakeLLMService, FakeMemoryStore, make_memory


@pytest.mark.asyncio
async def test_recall_returns_relevant_memories_and_reinforces_access() -> None:
    memories = [
        make_memory("Postgres pgvector was chosen for durable memory.", namespace="repo:backend"),
        make_memory("Temporary frontend color discussion.", namespace="repo:backend"),
        make_memory("SQLite belongs to another namespace.", namespace="repo:other"),
    ]
    memory_store = FakeMemoryStore(memories)
    graph_store = FakeGraphStore()
    service = RecallService(memory_store, graph_store, FakeLLMService())

    results = await service.recall("repo:backend", "Postgres pgvector memory", top_k=2)

    assert len(results) == 2
    assert results[0]["namespace"] == "repo:backend"
    assert "Postgres" in results[0]["raw_content"]
    assert {result["id"] for result in results} == set(memory_store.marked_accessed)
    assert all(memory.access_count == 1 for memory in memories[:2])
    assert len(graph_store.edges) == 1


@pytest.mark.asyncio
async def test_recall_with_random_synthetic_data_finds_planted_memory() -> None:
    rng = random.Random(42)
    topics = ["cache", "queue", "auth", "billing", "search", "metrics"]
    memories = [
        make_memory(
            f"{rng.choice(topics)} note {index} with token {rng.randint(1, 1000)}",
            namespace="synthetic",
        )
        for index in range(120)
    ]
    planted = make_memory(
        "durable architecture decision: use Postgres pgvector for cognitive memory",
        namespace="synthetic",
        base_importance=1.0,
    )
    memories.append(planted)

    service = RecallService(FakeMemoryStore(memories), FakeGraphStore(), FakeLLMService())

    results = await service.recall("synthetic", "Postgres pgvector cognitive memory", top_k=5)

    assert planted.id in {result["id"] for result in results}


@pytest.mark.performance
@pytest.mark.asyncio
async def test_fake_recall_performance_with_random_synthetic_data() -> None:
    rng = random.Random(7)
    memories = [
        make_memory(
            f"synthetic memory {index} project {rng.randint(1, 50)} decision {rng.random()}",
            namespace="perf",
        )
        for index in range(1_000)
    ]
    service = RecallService(FakeMemoryStore(memories), FakeGraphStore(), FakeLLMService())

    started = time.perf_counter()
    results = await service.recall("perf", "project decision 12", top_k=10)
    elapsed = time.perf_counter() - started

    assert len(results) == 10
    assert elapsed < 0.5
