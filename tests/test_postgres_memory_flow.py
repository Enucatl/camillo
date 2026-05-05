import os
import random
import time
from collections.abc import AsyncIterator
from typing import ClassVar
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.cognitive.ingestion_service import IngestionService
from camillo.cognitive.recall_service import RecallService
from camillo.db.models import HebbianEdge, Memory
from camillo.db.session import AsyncSessionLocal
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from tests.fakes import FakeLLMService

pytestmark = pytest.mark.integration


class KeywordLLMService:
    """Make database integration assertions semantic without external providers."""

    keywords: ClassVar[list[str]] = [
        "postgres",
        "pgvector",
        "database",
        "schema",
        "migration",
        "rollback",
        "alembic",
        "deploy",
        "canary",
        "healthcheck",
        "container",
        "docker",
        "cache",
        "redis",
        "queue",
        "latency",
        "garden",
        "tomato",
        "irrigation",
    ]

    def __init__(self, dim: int, valence: float = 0.7):
        """Configure deterministic provider behavior for PostgreSQL tests.

        Args:
            dim: Embedding dimension required by the configured database vector.
            valence: Importance score returned during ingestion.
        """
        self.dim = dim
        self.valence = valence
        self.scored: list[str] = []
        self.embedded: list[str] = []

    async def score_valence(self, user_msg: str, ai_msg: str) -> float:
        """Avoid remote completion calls while still exercising ingestion.

        Args:
            user_msg: User-side turn content.
            ai_msg: Assistant-side turn content.

        Returns:
            The configured deterministic valence score.
        """
        self.scored.append(f"User:\n{user_msg}\n\nAssistant:\n{ai_msg}")
        return self.valence

    async def get_embedding(self, text: str) -> list[float]:
        """Map known keywords to vector dimensions for stable recall assertions.

        Args:
            text: Memory or query text to embed.

        Returns:
            A deterministic keyword-count vector.
        """
        self.embedded.append(text)
        normalized = text.casefold()
        embedding = [0.0] * self.dim
        for index, keyword in enumerate(self.keywords):
            if index >= self.dim:
                break
            embedding[index] = float(normalized.count(keyword))
        if not any(embedding):
            embedding[-1] = 1.0
        return embedding

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Keep integration reranking deterministic and query-sensitive.

        Args:
            query: Recall query.
            documents: Candidate memory texts.

        Returns:
            Query-term overlap scores in document order.
        """
        query_terms = set(query.casefold().split())
        scores = []
        for document in documents:
            document_terms = set(document.casefold().split())
            scores.append(len(query_terms & document_terms) / max(len(query_terms), 1))
        return scores


def _db_tests_enabled() -> bool:
    """Keep PostgreSQL tests opt-in because they require external services.

    Returns:
        Whether database-backed tests should run in this process.
    """
    return os.getenv("RUN_DB_TESTS") == "1"


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Provide a rollback-scoped database session for integration tests.

    Yields:
        Async SQLAlchemy session connected to a migrated test database.
    """
    if not _db_tests_enabled():
        pytest.skip("Set RUN_DB_TESTS=1 with a migrated PostgreSQL/pgvector database to run")

    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.mark.asyncio
async def test_postgres_end_to_end_memory_recall_edges_and_reinforcement(
    db_session: AsyncSession,
) -> None:
    """Protect Phase 2 behavior against the real PostgreSQL stores."""
    namespace = f"test:{uuid4()}"
    memory_store = MemoryStore(db_session)
    graph_store = GraphStore(db_session)
    llm_service = KeywordLLMService(dim=settings.embedding_dim, valence=0.88)
    service = IngestionService(
        memory_store,
        graph_store,
        llm_service,
    )

    migration_plan = await service.ingest_interaction(
        namespace,
        "Postgres schema migration plan uses Alembic with a rollback checklist.",
        "Remember the database migration rollback plan for pgvector tables.",
        "schema-session-a",
    )
    migration_incident = await service.ingest_interaction(
        namespace,
        "Postgres database migration rehearsal found a schema rollback ordering issue.",
        "Remember that Alembic rollback must handle pgvector indexes after tables.",
        "schema-session-b",
    )
    deployment = await service.ingest_interaction(
        namespace,
        "Docker container deploy uses a canary rollout and healthcheck gate.",
        "Remember the deployment healthcheck and canary container procedure.",
        "deploy-session",
    )
    garden = await service.ingest_interaction(
        namespace,
        "Garden tomato irrigation notes belong to a backyard watering project.",
        "Remember the tomato irrigation schedule separately from engineering notes.",
        "garden-session",
    )
    await db_session.commit()

    stored = (
        (await db_session.execute(select(Memory).where(Memory.namespace == namespace)))
        .scalars()
        .all()
    )
    assert len(stored) == 4
    assert all(len(memory.embedding) == settings.embedding_dim for memory in stored)

    no_ingestion_edge = (
        await db_session.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id.in_([migration_plan.id, migration_incident.id]),
                HebbianEdge.target_id.in_([migration_plan.id, migration_incident.id]),
            )
        )
    ).scalar_one_or_none()
    assert no_ingestion_edge is None

    recall_service = RecallService(
        memory_store,
        graph_store,
        llm_service,
    )
    missing_namespace_results = await recall_service.recall(
        f"missing:{uuid4()}",
        "Postgres schema migration rollback",
        top_k=3,
    )
    assert missing_namespace_results == []

    migration_results = await recall_service.recall(
        namespace,
        "Postgres schema migration rollback database pgvector",
        top_k=2,
    )
    await db_session.commit()

    migration_result_ids = {result.memory.id for result in migration_results}
    assert migration_result_ids == {migration_plan.id, migration_incident.id}
    assert deployment.id not in migration_result_ids
    assert garden.id not in migration_result_ids

    created_edge = (
        await db_session.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id.in_([migration_plan.id, migration_incident.id]),
                HebbianEdge.target_id.in_([migration_plan.id, migration_incident.id]),
            )
        )
    ).scalar_one()
    assert created_edge.weight == 1.0

    refreshed = (
        (
            await db_session.execute(
                select(Memory).where(Memory.id.in_([migration_plan.id, migration_incident.id]))
            )
        )
        .scalars()
        .all()
    )
    assert all(memory.access_count == 1 for memory in refreshed)
    assert all(memory.last_accessed_at >= memory.created_at for memory in refreshed)

    deployment_results = await recall_service.recall(
        namespace,
        "docker container canary deploy healthcheck",
        top_k=1,
    )
    await db_session.commit()
    assert [result.memory.id for result in deployment_results] == [deployment.id]

    second_migration_results = await recall_service.recall(
        namespace,
        "alembic database schema migration rollback",
        top_k=2,
    )
    await db_session.commit()
    assert {result.memory.id for result in second_migration_results} == {
        migration_plan.id,
        migration_incident.id,
    }

    reinforced_edge = (
        await db_session.execute(
            select(HebbianEdge).where(
                HebbianEdge.source_id == created_edge.source_id,
                HebbianEdge.target_id == created_edge.target_id,
            )
        )
    ).scalar_one()
    assert reinforced_edge.weight == 2.0

    reinforced_memories = (
        (
            await db_session.execute(
                select(Memory).where(Memory.id.in_([migration_plan.id, migration_incident.id]))
            )
        )
        .scalars()
        .all()
    )
    assert all(memory.access_count == 2 for memory in reinforced_memories)

    deployment_memory = (
        await db_session.execute(select(Memory).where(Memory.id == deployment.id))
    ).scalar_one()
    assert deployment_memory.access_count == 1

    await db_session.execute(delete(Memory).where(Memory.namespace == namespace))
    await db_session.commit()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_postgres_synthetic_recall_performance(
    db_session: AsyncSession,
) -> None:
    """Protect database recall from obvious performance regressions when opted in."""
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
