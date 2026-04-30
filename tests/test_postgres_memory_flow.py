import os
import random
import time
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from camillo.cognitive.ingestion_service import IngestionService
from camillo.cognitive.recall_service import RecallService
from camillo.db.models import HebbianEdge, Memory
from camillo.db.session import AsyncSessionLocal
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from tests.fakes import FakeLLMService

pytestmark = pytest.mark.integration


def _db_tests_enabled() -> bool:
    return os.getenv("RUN_DB_TESTS") == "1"


@pytest.fixture
async def db_session():
    if not _db_tests_enabled():
        pytest.skip("Set RUN_DB_TESTS=1 with a migrated PostgreSQL/pgvector database to run")

    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.mark.asyncio
async def test_postgres_ingest_recall_edges_and_reinforcement(db_session) -> None:
    namespace = f"test:{uuid4()}"
    memory_store = MemoryStore(db_session)
    graph_store = GraphStore(db_session)
    service = IngestionService(
        memory_store,
        graph_store,
        FakeLLMService(dim=settings.embedding_dim, valence=0.88),
    )

    first = await service.ingest_interaction(
        namespace,
        "We chose Postgres with pgvector for cognitive memory.",
        "I will remember the Postgres-native architecture.",
        "session-a",
    )
    second = await service.ingest_interaction(
        namespace,
        "Adjacent turns in the same session should form Hebbian edges.",
        "I will reinforce session adjacency.",
        "session-a",
    )
    await db_session.commit()

    stored = (
        (await db_session.execute(select(Memory).where(Memory.namespace == namespace)))
        .scalars()
        .all()
    )
    assert len(stored) == 2
    assert all(len(memory.embedding) == settings.embedding_dim for memory in stored)

    edge = (
        await db_session.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id.in_([first.id, second.id]),
                HebbianEdge.target_id.in_([first.id, second.id]),
            )
        )
    ).scalar_one()
    assert edge.weight == 1.0

    recall_service = RecallService(
        memory_store,
        graph_store,
        FakeLLMService(dim=settings.embedding_dim),
    )
    results = await recall_service.recall(namespace, "Postgres pgvector architecture", top_k=2)
    await db_session.commit()

    assert len(results) == 2
    assert first.id in {result["id"] for result in results}

    refreshed = (
        (await db_session.execute(select(Memory).where(Memory.id.in_([first.id, second.id]))))
        .scalars()
        .all()
    )
    assert all(memory.access_count >= 1 for memory in refreshed)

    reinforced_edge = (
        await db_session.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id == edge.source_id,
                HebbianEdge.target_id == edge.target_id,
            )
        )
    ).scalar_one()
    assert reinforced_edge.weight >= 2.0

    await db_session.execute(delete(Memory).where(Memory.namespace == namespace))
    await db_session.commit()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_postgres_synthetic_recall_performance(db_session) -> None:
    if os.getenv("RUN_PERF_TESTS") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run PostgreSQL performance checks")

    namespace = f"perf:{uuid4()}"
    rng = random.Random(99)
    memory_store = MemoryStore(db_session)
    graph_store = GraphStore(db_session)
    llm_service = FakeLLMService(dim=settings.embedding_dim, valence=0.6)

    for index in range(200):
        await memory_store.insert_memory(
            namespace=namespace,
            raw_content=f"synthetic memory {index} topic {rng.randint(1, 25)} pgvector recall",
            embedding=llm_service.dim * [rng.random()],
            memory_type="episodic",
            base_importance=rng.uniform(0.2, 1.0),
            session_id="perf-session",
        )
    await db_session.commit()

    recall_service = RecallService(memory_store, graph_store, llm_service)
    started = time.perf_counter()
    results = await recall_service.recall(namespace, "pgvector recall topic 7", top_k=10)
    elapsed = time.perf_counter() - started
    await db_session.commit()

    assert len(results) == 10
    assert elapsed < 2.0

    await db_session.execute(delete(Memory).where(Memory.namespace == namespace))
    await db_session.commit()
