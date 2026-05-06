import random
import time

import pytest

from camillo.cognitive.recall_service import RecallService
from tests.fakes import FakeGraphStore, FakeLLMService, FakeMemoryStore, make_memory


@pytest.mark.asyncio
async def test_recall_returns_relevant_memories_and_reinforces_access() -> None:
    """Protect Phase 1 compatibility while verifying Phase 2 reinforcement."""
    memories = [
        make_memory("Postgres pgvector was chosen for durable memory.", namespace="repo:backend"),
        make_memory("Temporary Postgres pgvector memory discussion.", namespace="repo:backend"),
        make_memory("SQLite belongs to another namespace.", namespace="repo:other"),
    ]
    memory_store = FakeMemoryStore(memories)
    graph_store = FakeGraphStore()
    service = RecallService(memory_store, graph_store, FakeLLMService())

    results = await service.recall("repo:backend", "Postgres pgvector memory", top_k=2)

    assert len(results) == 2
    assert results[0].memory.namespace == "repo:backend"
    assert "Postgres" in results[0].memory.raw_content
    assert {result.memory.id for result in results} == set(memory_store.marked_accessed)
    assert all(memory.access_count == 1 for memory in memories[:2])
    assert len(graph_store.edges) == 1


@pytest.mark.asyncio
async def test_recall_with_random_synthetic_data_finds_planted_memory() -> None:
    """Protect recall quality against noisy candidate pools."""
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

    assert planted.id in {result.memory.id for result in results}


@pytest.mark.performance
@pytest.mark.asyncio
async def test_fake_recall_performance_with_random_synthetic_data() -> None:
    """Protect the fake pipeline from accidental quadratic behavior."""
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


@pytest.mark.asyncio
async def test_recall_pipeline_uses_rerank_activation_and_reinforcement() -> None:
    """Protect the intended order of rerank, activation scoring, and access updates."""
    memories = [
        make_memory("Postgres pgvector durable memory.", namespace="repo"),
        make_memory("FastAPI service recall pipeline.", namespace="repo"),
    ]
    llm_service = FakeLLMService()
    memory_store = FakeMemoryStore(memories)
    graph_store = FakeGraphStore()
    service = RecallService(memory_store, graph_store, llm_service)

    results = await service.recall("repo", "Postgres recall", top_k=1, include_hebbian=False)

    assert llm_service.embedded == ["Postgres recall"]
    assert llm_service.reranked
    assert len(results) == 1
    assert results[0].activation_score is not None
    assert results[0].final_score is not None
    assert memory_store.marked_accessed == [results[0].memory.id]
    assert graph_store.edges == {}


@pytest.mark.asyncio
async def test_recall_drops_low_rerank_scores() -> None:
    """Protect the relevance threshold so weak reranked memories are filtered."""

    class LowScoreLLM(FakeLLMService):
        """Force low relevance to isolate threshold behavior."""

        async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
            """Return zero scores so the test is independent of token overlap."""
            self.reranked.append((query, documents))
            return [0.0 for _ in documents]

    memories = [make_memory("Postgres pgvector durable memory.", namespace="repo")]
    service = RecallService(FakeMemoryStore(memories), FakeGraphStore(), LowScoreLLM())

    results = await service.recall("repo", "Postgres", top_k=1, include_hebbian=False)

    assert results == []


@pytest.mark.asyncio
async def test_recall_adds_hebbian_neighbors_after_primary_results() -> None:
    """Protect the rule that graph context is appended after primary recall."""
    primary = make_memory("Postgres pgvector durable memory.", namespace="repo")
    neighbor = make_memory("Alembic migration context.", namespace="repo:linked", scope="shared")
    llm_service = FakeLLMService()
    memory_store = FakeMemoryStore([primary, neighbor])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(primary.id, neighbor.id, increment=3.0)
    service = RecallService(memory_store, graph_store, llm_service)

    results = await service.recall("repo", "Postgres", top_k=1, include_hebbian=True)

    assert [result.source for result in results] == ["primary", "hebbian"]
    assert results[1].memory.id == neighbor.id
    assert results[1].linked_from == primary.id
    assert results[1].edge_weight == 3.0


@pytest.mark.asyncio
async def test_recall_includes_shared_and_global_memories_by_default() -> None:
    """Allow reusable memories to cross namespace boundaries."""
    local = make_memory("Postgres pgvector local memory.", namespace="repo")
    shared = make_memory("Postgres pgvector shared procedure.", namespace="other", scope="shared")
    global_memory = make_memory(
        "Postgres pgvector global preference.",
        namespace="user:default",
        scope="global",
    )
    excluded = make_memory("Postgres pgvector private note.", namespace="other", scope="local")
    memory_store = FakeMemoryStore([local, shared, global_memory, excluded])
    service = RecallService(memory_store, FakeGraphStore(), FakeLLMService())

    results = await service.recall("repo", "Postgres pgvector", top_k=10, include_hebbian=False)

    result_ids = {result.memory.id for result in results}
    assert local.id in result_ids
    assert shared.id in result_ids
    assert global_memory.id in result_ids
    assert excluded.id not in result_ids


@pytest.mark.asyncio
async def test_recall_can_disable_shared_memories() -> None:
    """Preserve strict namespace-local recall when requested."""
    local = make_memory("Postgres pgvector local memory.", namespace="repo")
    shared = make_memory("Postgres pgvector shared procedure.", namespace="other", scope="shared")
    memory_store = FakeMemoryStore([local, shared])
    service = RecallService(memory_store, FakeGraphStore(), FakeLLMService())

    results = await service.recall(
        "repo",
        "Postgres pgvector",
        top_k=10,
        include_hebbian=False,
        include_shared=False,
    )

    assert [result.memory.id for result in results] == [local.id]
