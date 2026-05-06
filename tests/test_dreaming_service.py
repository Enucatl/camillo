from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from camillo.cognitive.dreaming_service import DreamingService
from camillo.db.models import DreamRun, MemoryRelation
from camillo.schemas.submit_memory import MemorySubmissionReport
from camillo.settings import settings
from tests.fakes import FakeGraphStore, FakeMemoryStore, make_memory


class FakeDreamStore:
    """Capture dream run bookkeeping without a database."""

    def __init__(self):
        """Start with no dream runs."""
        self.runs: dict[UUID, DreamRun] = {}

    async def create_run(
        self,
        namespace: str,
        *,
        dry_run: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DreamRun:
        """Create a model-shaped fake dream run."""
        run = DreamRun(
            id=uuid4(),
            namespace=namespace,
            status="dry_run" if dry_run else "running",
            started_at=datetime.now(UTC),
            metadata_json=metadata or {},
        )
        self.runs[run.id] = run
        return run

    async def complete_run(
        self,
        dream_run_id: UUID,
        *,
        seed_memory_ids: list[UUID],
        source_memory_ids: list[UUID],
        created_memory_ids: list[UUID],
        clusters_considered: int,
        clusters_dreamed: int,
        memories_created: int,
        dry_run: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DreamRun:
        """Store completion counters for assertions."""
        run = self.runs[dream_run_id]
        run.status = "dry_run" if dry_run else "completed"
        run.completed_at = datetime.now(UTC)
        run.seed_memory_ids = seed_memory_ids
        run.source_memory_ids = source_memory_ids
        run.created_memory_ids = created_memory_ids
        run.clusters_considered = clusters_considered
        run.clusters_dreamed = clusters_dreamed
        run.memories_created = memories_created
        return run

    async def fail_run(self, dream_run_id: UUID, error: str) -> DreamRun:
        """Store failure details for parity with production store."""
        run = self.runs[dream_run_id]
        run.status = "failed"
        run.completed_at = datetime.now(UTC)
        run.error = error
        return run


class FakeRelationStore:
    """Capture consolidation relations."""

    def __init__(self):
        """Start with no captured relations."""
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
        """Append a model-shaped relation."""
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


class FakeDreamLLM:
    """Return caller-configured dream synthesis outputs."""

    def __init__(self, dream: dict[str, Any]):
        """Capture the synthetic dream response."""
        self.dream = dream
        self.calls: list[list[str]] = []

    async def synthesize_dream(self, cluster_memories: list[str], *, namespace: str) -> dict:
        """Return the configured dream and record source content."""
        self.calls.append(cluster_memories)
        return self.dream


class FakeReconciliationService:
    """Capture dreamed memory submissions."""

    def __init__(self, outcome: str = "created", memory_id: UUID | None = None):
        """Configure the reconciliation outcome."""
        self.outcome = outcome
        self.memory_id = memory_id or uuid4()
        self.submissions: list[dict[str, Any]] = []

    async def submit_memory(self, **kwargs: Any) -> MemorySubmissionReport:
        """Return a configured successful submission report."""
        self.submissions.append(kwargs)
        if self.outcome == "created":
            return MemorySubmissionReport(
                outcome="created",
                created_memory_id=self.memory_id,
                message="created",
            )
        return MemorySubmissionReport(
            outcome="ignored_duplicate",
            affected_memory_ids=[self.memory_id],
            message="reinforced",
        )


@pytest.fixture(autouse=True)
def dreaming_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use permissive, deterministic dreaming settings in unit tests."""
    monkeypatch.setattr(settings, "dreaming_min_seed_activation", 0.0)
    monkeypatch.setattr(settings, "dreaming_seed_limit", 5)
    monkeypatch.setattr(settings, "dreaming_cluster_max_size", 12)
    monkeypatch.setattr(settings, "dreaming_cluster_min_size", 2)
    monkeypatch.setattr(settings, "dreaming_max_depth", 2)
    monkeypatch.setattr(settings, "dreaming_min_edge_weight", 2.0)
    monkeypatch.setattr(settings, "dreaming_max_cluster_age_days", 90)
    monkeypatch.setattr(settings, "dreaming_min_cluster_total_importance", 0.0)
    monkeypatch.setattr(settings, "dreaming_min_synthesis_confidence", 0.6)
    monkeypatch.setattr(settings, "dreaming_max_memories_per_cluster", 3)
    monkeypatch.setattr(settings, "dreaming_source_penalty", 0.35)
    monkeypatch.setattr(settings, "dreaming_min_source_importance", 0.05)
    monkeypatch.setattr(settings, "dreaming_relation_confidence", 0.85)


def _service(
    memory_store: FakeMemoryStore,
    graph_store: FakeGraphStore,
    llm_service: FakeDreamLLM,
    reconciliation_service: FakeReconciliationService | None = None,
    relation_store: FakeRelationStore | None = None,
) -> tuple[DreamingService, FakeRelationStore, FakeReconciliationService]:
    """Build a dreaming service with fake collaborators."""
    fake_relation_store = relation_store or FakeRelationStore()
    fake_reconciliation = reconciliation_service or FakeReconciliationService()
    service = DreamingService(
        memory_store,
        graph_store,
        fake_relation_store,  # type: ignore[arg-type]
        FakeDreamStore(),  # type: ignore[arg-type]
        fake_reconciliation,  # type: ignore[arg-type]
        llm_service,  # type: ignore[arg-type]
    )
    return service, fake_relation_store, fake_reconciliation


@pytest.mark.asyncio
async def test_select_dream_seeds_returns_only_active_episodic_memories() -> None:
    """Protect the primary anti-repeat seed filter."""
    active_episodic = make_memory("active episodic", namespace="repo")
    semantic = make_memory("semantic", namespace="repo", memory_type="semantic")
    other_namespace = make_memory("other", namespace="other")
    memory_store = FakeMemoryStore([active_episodic, semantic, other_namespace])

    seeds = await memory_store.select_dream_seeds(
        "repo",
        limit=10,
        min_activation=0.0,
        decay_rate=0.01,
    )

    assert [memory.id for memory in seeds] == [active_episodic.id]


@pytest.mark.asyncio
async def test_select_dream_seeds_excludes_consolidated_episodic_memories() -> None:
    """Ensure consolidated episodes cannot seed future dreams."""
    active = make_memory("active", namespace="repo")
    consolidated = make_memory("consolidated", namespace="repo")
    consolidated.status = "consolidated"
    memory_store = FakeMemoryStore([active, consolidated])

    seeds = await memory_store.select_dream_seeds(
        "repo",
        limit=10,
        min_activation=0.0,
        decay_rate=0.01,
    )

    assert [memory.id for memory in seeds] == [active.id]


@pytest.mark.asyncio
async def test_select_dream_seeds_sorts_by_activation() -> None:
    """Keep seed order activation-driven."""
    low = make_memory("low", namespace="repo", base_importance=0.1)
    high = make_memory("high", namespace="repo", base_importance=0.9)
    memory_store = FakeMemoryStore([low, high])

    seeds = await memory_store.select_dream_seeds(
        "repo",
        limit=10,
        min_activation=0.0,
        decay_rate=0.01,
    )

    assert [memory.id for memory in seeds] == [high.id, low.id]


@pytest.mark.asyncio
async def test_traverse_hebbian_cluster_respects_depth_weight_and_node_limit() -> None:
    """Protect graph traversal bounds before database-specific tests."""
    first = make_memory("first")
    second = make_memory("second")
    third = make_memory("third")
    weak = make_memory("weak")
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    await graph_store.create_or_increment_edge(second.id, third.id, 3.0)
    await graph_store.create_or_increment_edge(first.id, weak.id, 1.0)

    depth_one = await graph_store.traverse_hebbian_cluster(
        first.id,
        max_depth=1,
        min_weight=2.0,
        max_nodes=10,
    )
    capped = await graph_store.traverse_hebbian_cluster(
        first.id,
        max_depth=2,
        min_weight=2.0,
        max_nodes=2,
    )

    assert [item[0] for item in depth_one] == [first.id, second.id]
    assert [item[0] for item in capped] == [first.id, second.id]


@pytest.mark.asyncio
async def test_dreaming_service_skips_clusters_below_min_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not synthesize clusters that lack enough active episodic sources."""
    monkeypatch.setattr(settings, "dreaming_cluster_min_size", 3)
    first = make_memory("first", namespace="repo")
    second = make_memory("second", namespace="repo")
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    service, _relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM({"should_create_memory": True, "summary": "", "memories": []}),
    )

    report = await service.run_once("repo")

    assert report.clusters_considered == 0
    assert reconciliation.submissions == []


@pytest.mark.asyncio
async def test_dreaming_service_skips_clusters_below_total_importance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not dream low-importance clusters."""
    monkeypatch.setattr(settings, "dreaming_min_cluster_total_importance", 2.0)
    first = make_memory("first", namespace="repo", base_importance=0.2)
    second = make_memory("second", namespace="repo", base_importance=0.2)
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    service, _relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM({"should_create_memory": True, "summary": "", "memories": []}),
    )

    report = await service.run_once("repo")

    assert report.clusters_considered == 0
    assert reconciliation.submissions == []


@pytest.mark.asyncio
async def test_dreaming_service_deduplicates_overlapping_clusters_within_run() -> None:
    """Avoid dreaming the same connected component twice in one run."""
    memories = [make_memory(f"memory {index}", namespace="repo") for index in range(3)]
    memory_store = FakeMemoryStore(memories)
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(memories[0].id, memories[1].id, 3.0)
    await graph_store.create_or_increment_edge(memories[1].id, memories[2].id, 3.0)
    dream = {
        "should_create_memory": True,
        "summary": "durable",
        "memories": [
            {
                "content": "Durable memory.",
                "memory_type": "semantic",
                "confidence": 0.9,
                "evidence_indices": [0, 1],
                "rationale": "supported",
            }
        ],
    }
    service, _relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM(dream),
    )

    report = await service.run_once("repo")

    assert report.clusters_considered == 1
    assert len(reconciliation.submissions) == 1


@pytest.mark.asyncio
async def test_llm_noop_dream_does_not_consolidate_sources() -> None:
    """A no-op synthesis should leave source episodes active."""
    first = make_memory("first", namespace="repo")
    second = make_memory("second", namespace="repo")
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    service, _relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM({"should_create_memory": False, "summary": "none", "memories": []}),
    )

    report = await service.run_once("repo")

    assert report.memories_created == 0
    assert reconciliation.submissions == []
    assert first.status == "active"
    assert second.status == "active"


@pytest.mark.asyncio
async def test_successful_dream_submits_memory_and_creates_relations() -> None:
    """Successful consolidation must flow through reconciliation and relations."""
    first = make_memory("first", namespace="repo")
    second = make_memory("second", namespace="repo")
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    semantic_id = uuid4()
    dream = {
        "should_create_memory": True,
        "summary": "durable",
        "memories": [
            {
                "content": "Durable memory.",
                "memory_type": "semantic",
                "confidence": 0.9,
                "evidence_indices": [0, 1],
                "rationale": "supported",
            }
        ],
    }
    service, relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM(dream),
        FakeReconciliationService(memory_id=semantic_id),
    )

    report = await service.run_once("repo")

    assert report.memories_created == 1
    assert reconciliation.submissions[0]["content"] == "Durable memory."
    assert {relation.target_id for relation in relations.relations} == {first.id, second.id}
    assert all(relation.source_id == semantic_id for relation in relations.relations)
    assert all(relation.relation_type == "consolidates" for relation in relations.relations)


@pytest.mark.asyncio
async def test_successful_dream_marks_sources_consolidated() -> None:
    """Successful semantic promotion should change source memory state."""
    first = make_memory("first", namespace="repo", base_importance=0.8)
    second = make_memory("second", namespace="repo", base_importance=0.8)
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    dream = {
        "should_create_memory": True,
        "summary": "durable",
        "memories": [
            {
                "content": "Durable memory.",
                "memory_type": "semantic",
                "confidence": 0.9,
                "evidence_indices": [0, 1],
                "rationale": "supported",
            }
        ],
    }
    service, _relations, _reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM(dream),
    )

    await service.run_once("repo")

    assert first.status == "consolidated"
    assert second.status == "consolidated"
    assert first.metadata_json["dreaming"]["consolidated_into"]


@pytest.mark.asyncio
async def test_dry_run_performs_no_writes_except_dream_bookkeeping() -> None:
    """Dry runs should report proposals without reconciliation or consolidation."""
    first = make_memory("first", namespace="repo")
    second = make_memory("second", namespace="repo")
    memory_store = FakeMemoryStore([first, second])
    graph_store = FakeGraphStore()
    await graph_store.create_or_increment_edge(first.id, second.id, 3.0)
    dream = {
        "should_create_memory": True,
        "summary": "durable",
        "memories": [
            {
                "content": "Durable memory.",
                "memory_type": "semantic",
                "confidence": 0.9,
                "evidence_indices": [0, 1],
                "rationale": "supported",
            }
        ],
    }
    service, relations, reconciliation = _service(
        memory_store,
        graph_store,
        FakeDreamLLM(dream),
    )

    report = await service.run_once("repo", dry_run=True)

    assert report.status == "dry_run"
    assert report.clusters[0].dreamed_memories[0].outcome == "dry_run"
    assert reconciliation.submissions == []
    assert relations.relations == []
    assert first.status == "active"
    assert second.status == "active"
