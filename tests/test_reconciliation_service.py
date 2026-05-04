from typing import Any
from uuid import UUID

import pytest

from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.models import MemoryRelation
from camillo.schemas.submit_memory import MemoryRelationshipClassification
from tests.fakes import FakeGraphStore, FakeLLMService, FakeMemoryStore, make_memory


class FakeRelationStore:
    """Capture semantic relations without a database.

    Reconciliation tests need to observe policy decisions, not SQLAlchemy
    upsert behavior, so this fake keeps the relation contract in memory.
    """

    def __init__(self):
        """Start with no semantic relations."""
        self.relations: list[MemoryRelation] = []

    async def create_relation(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        confidence: float = 0.8,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRelation:
        """Append a model-shaped relation for assertions."""
        relation = MemoryRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            rationale=rationale,
            metadata_json=metadata or {},
        )
        self.relations.append(relation)
        return relation

    async def get_relations_for_memory(self, memory_id: UUID) -> list[MemoryRelation]:
        """Return fake relations touching the requested memory."""
        return [
            relation
            for relation in self.relations
            if relation.source_id == memory_id or relation.target_id == memory_id
        ]


def _service(
    memory_store: FakeMemoryStore,
    llm_service: FakeLLMService,
    relation_store: FakeRelationStore | None = None,
) -> MemoryReconciliationService:
    """Build the real reconciliation service around fake dependencies."""
    graph_store = FakeGraphStore()
    recall_service = RecallService(memory_store, graph_store, llm_service)
    return MemoryReconciliationService(
        memory_store,
        relation_store or FakeRelationStore(),
        recall_service,
        llm_service,
    )


@pytest.mark.asyncio
async def test_submit_memory_creates_new_memory_when_no_related_memories_exist() -> None:
    """Protect new-memory creation when recall finds no active context."""
    memory_store = FakeMemoryStore()
    llm_service = FakeLLMService()
    service = _service(memory_store, llm_service)

    report = await service.submit_memory("repo", "Use pgvector for semantic memory.")

    assert report.outcome == "created"
    assert report.created_memory_id is not None
    assert len(memory_store.memories) == 1
    assert memory_store.memories[0].type == "semantic"


@pytest.mark.asyncio
async def test_submit_memory_reinforces_duplicate_instead_of_creating_memory() -> None:
    """Avoid duplicate durable rows for high-confidence duplicate judgments."""
    old_memory = make_memory("Use pgvector for semantic memory.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    llm_service = FakeLLMService()
    llm_service.classifications = [
        MemoryRelationshipClassification(index=0, relation="duplicate", confidence=0.91)
    ]
    service = _service(memory_store, llm_service)

    report = await service.submit_memory("repo", "Use pgvector for semantic memory.")

    assert report.outcome == "ignored_duplicate"
    assert report.affected_memory_ids == [old_memory.id]
    assert len(memory_store.memories) == 1
    assert old_memory.access_count >= 1


@pytest.mark.asyncio
async def test_submit_memory_supersedes_only_when_resolution_supports_supersession() -> None:
    """Require explicit supersession policy before marking old memory inactive."""
    old_memory = make_memory("The project uses Qdrant for vectors.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    relation_store = FakeRelationStore()
    llm_service = FakeLLMService()
    llm_service.classifications = [
        MemoryRelationshipClassification(
            index=0,
            relation="contradicts",
            confidence=0.9,
            contradiction_type="implementation_change",
            resolution="supersede_old",
        )
    ]
    service = _service(memory_store, llm_service, relation_store)

    report = await service.submit_memory("repo", "The project now uses pgvector for vectors.")

    assert report.outcome == "superseded_old_memory"
    assert old_memory.status == "superseded"
    assert old_memory.superseded_by == report.created_memory_id
    assert relation_store.relations[0].relation_type == "supersedes"


@pytest.mark.asyncio
async def test_submit_memory_keeps_both_for_contextual_contradiction() -> None:
    """Keep contextual contradictions active instead of replacing old context."""
    old_memory = make_memory("Production caching uses Redis.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    relation_store = FakeRelationStore()
    llm_service = FakeLLMService()
    llm_service.classifications = [
        MemoryRelationshipClassification(
            index=0,
            relation="contradicts",
            confidence=0.88,
            contradiction_type="environment_difference",
            resolution="keep_both",
        )
    ]
    service = _service(memory_store, llm_service, relation_store)

    report = await service.submit_memory("repo", "Local tests use in-memory caching.")

    assert report.outcome == "created"
    assert old_memory.status == "active"
    assert len(memory_store.memories) == 2
    assert relation_store.relations[0].relation_type == "contradicts"


@pytest.mark.asyncio
async def test_submit_memory_creates_exception_relation() -> None:
    """Persist exception_to when contradiction resolution asks for an exception."""
    old_memory = make_memory("All API routes require auth.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    relation_store = FakeRelationStore()
    llm_service = FakeLLMService()
    llm_service.classifications = [
        MemoryRelationshipClassification(
            index=0,
            relation="contradicts",
            confidence=0.82,
            resolution="create_exception",
        )
    ]
    service = _service(memory_store, llm_service, relation_store)

    report = await service.submit_memory("repo", "The API health route has public auth.")

    assert report.outcome == "created"
    assert relation_store.relations[0].relation_type == "exception_to"


@pytest.mark.asyncio
async def test_submit_memory_creates_refines_relation() -> None:
    """Persist refines when the new memory narrows older broad guidance."""
    old_memory = make_memory("Run pytest before commits.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    relation_store = FakeRelationStore()
    llm_service = FakeLLMService()
    llm_service.classifications = [
        MemoryRelationshipClassification(
            index=0,
            relation="contradicts",
            confidence=0.82,
            resolution="refine_old",
        )
    ]
    service = _service(memory_store, llm_service, relation_store)

    await service.submit_memory("repo", "Run containerized pytest before Docker changes.")

    assert relation_store.relations[0].relation_type == "refines"


@pytest.mark.asyncio
async def test_submit_memory_deprecates_related_memory_when_intent_is_forget() -> None:
    """Let explicit forget requests deprecate related active memories."""
    old_memory = make_memory("Remember the legacy Redis migration note.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    llm_service = FakeLLMService()
    service = _service(memory_store, llm_service)

    report = await service.submit_memory("repo", "legacy Redis migration", intent="forget")

    assert report.outcome == "deprecated_old_memory"
    assert old_memory.status == "deprecated"
    assert report.created_memory_id is None


@pytest.mark.asyncio
async def test_classifier_failure_fallback_creates_memory() -> None:
    """Fallback unrelated classifications should preserve submission progress."""

    class FailingLLM(FakeLLMService):
        """Raise from classification while keeping embeddings available."""

        async def classify_memory_relationships(
            self,
            intent: str,
            new_content: str,
            existing_memories: list,
        ) -> list[MemoryRelationshipClassification]:
            """Simulate classifier outage."""
            raise RuntimeError("classifier unavailable")

    old_memory = make_memory("Use Redis for caching.", namespace="repo")
    memory_store = FakeMemoryStore([old_memory])
    llm_service = FailingLLM()
    service = _service(memory_store, llm_service)

    report = await service.submit_memory("repo", "Use in-memory cache in tests.")

    assert report.outcome == "created"
    assert old_memory.status == "active"
