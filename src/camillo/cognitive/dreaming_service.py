from typing import Any
from uuid import UUID

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.models import Memory
from camillo.schemas.dreaming import DreamClusterReport, DreamedMemoryReport, DreamRunReport
from camillo.settings import settings
from camillo.stores.dream_store import DreamStore
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore

ALLOWED_DREAM_MEMORY_TYPES = {
    "semantic",
    "preference",
    "procedural",
    "relationship",
    "profile",
    "core",
}
SUCCESSFUL_CONSOLIDATION_OUTCOMES = {
    "created",
    "ignored_duplicate",
}


class DreamingService:
    """Promote graph-connected episodic memories into durable memories.

    The service keeps anti-repeat behavior simple by marking source episodes
    consolidated only after reconciliation successfully creates or reinforces a
    semantic memory.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        graph_store: GraphStore,
        relation_store: RelationStore,
        dream_store: DreamStore,
        reconciliation_service: MemoryReconciliationService,
        llm_service: LiteLLMService,
    ):
        """Wire dreaming to existing stores and reconciliation policy.

        Args:
            memory_store: Memory persistence and lifecycle boundary.
            graph_store: Hebbian traversal boundary.
            relation_store: Semantic relation persistence boundary.
            dream_store: Dream run audit boundary.
            reconciliation_service: Durable memory submission policy.
            llm_service: Provider used for semantic synthesis.
        """
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.relation_store = relation_store
        self.dream_store = dream_store
        self.reconciliation_service = reconciliation_service
        self.llm_service = llm_service

    async def run_once(
        self,
        namespace: str,
        *,
        seed_limit: int | None = None,
        dry_run: bool | None = None,
    ) -> DreamRunReport:
        """Run one dreaming pass for active episodic memories.

        Args:
            namespace: Memory partition to consolidate.
            seed_limit: Optional override for seed selection count.
            dry_run: Optional override to suppress consolidation side effects.

        Returns:
            A serializable report for worker logs or HTTP callers.
        """
        effective_dry_run = settings.dreaming_dry_run if dry_run is None else dry_run
        effective_seed_limit = seed_limit or settings.dreaming_seed_limit
        dream_run = await self.dream_store.create_run(
            namespace=namespace,
            dry_run=effective_dry_run,
            metadata={
                "seed_limit": effective_seed_limit,
                "cluster_max_size": settings.dreaming_cluster_max_size,
                "max_depth": settings.dreaming_max_depth,
            },
        )

        try:
            return await self._run_with_created_run(
                namespace,
                dream_run.id,
                effective_seed_limit,
                effective_dry_run,
                dream_run.started_at,
            )
        except Exception as exc:
            await self.dream_store.fail_run(dream_run.id, str(exc))
            raise

    async def _run_with_created_run(
        self,
        namespace: str,
        dream_run_id: UUID,
        seed_limit: int,
        dry_run: bool,
        started_at,
    ) -> DreamRunReport:
        """Execute dreaming after audit row creation.

        Args:
            namespace: Memory partition to consolidate.
            dream_run_id: Audit row ID.
            seed_limit: Effective seed cap.
            dry_run: Whether to suppress consolidation side effects.
            started_at: Start timestamp from the audit row.

        Returns:
            Completed run report.
        """
        seeds = await self.memory_store.select_dream_seeds(
            namespace=namespace,
            limit=seed_limit,
            min_activation=settings.dreaming_min_seed_activation,
            decay_rate=settings.decay_rate,
            max_age_days=settings.dreaming_max_cluster_age_days,
        )

        processed_clusters: list[set[UUID]] = []
        cluster_reports: list[DreamClusterReport] = []
        seed_memory_ids = [seed.id for seed in seeds]
        source_memory_ids: list[UUID] = []
        created_memory_ids: list[UUID] = []
        clusters_considered = 0
        clusters_dreamed = 0
        memories_created = 0

        for seed in seeds:
            cluster = await self._build_cluster(namespace, seed)
            if cluster is None:
                continue

            cluster_ids = {memory.id for memory in cluster}
            if _is_duplicate_in_current_run(cluster_ids, processed_clusters):
                continue

            processed_clusters.append(cluster_ids)
            clusters_considered += 1
            source_ids = [memory.id for memory in cluster]
            source_memory_ids.extend(source_ids)
            cluster_report = DreamClusterReport(
                seed_memory_id=seed.id,
                source_memory_ids=source_ids,
            )
            cluster_reports.append(cluster_report)

            dream = await self.llm_service.synthesize_dream(
                [memory.raw_content for memory in cluster],
                namespace=namespace,
            )
            cluster_report.summary = str(dream.get("summary") or "")
            if not dream.get("should_create_memory"):
                continue

            proposals = _validated_proposals(dream.get("memories"), len(cluster))
            if not proposals:
                continue

            clusters_dreamed += 1
            if dry_run:
                cluster_report.dreamed_memories.extend(
                    DreamedMemoryReport(
                        content=proposal["content"],
                        memory_type=proposal["memory_type"],
                        confidence=proposal["confidence"],
                        outcome="dry_run",
                        source_memory_ids=_evidence_source_ids(proposal, source_ids),
                    )
                    for proposal in proposals
                )
                continue

            semantic_ids_for_cluster: list[UUID] = []
            for proposal in proposals:
                report = await self.reconciliation_service.submit_memory(
                    namespace=namespace,
                    content=proposal["content"],
                    intent="remember",
                    memory_type=proposal["memory_type"],
                    evidence=(
                        f"Dream run {dream_run_id}; synthesized from episodic memories: "
                        f"{[str(memory_id) for memory_id in source_ids]}"
                    ),
                    confidence=proposal["confidence"],
                )
                semantic_id = _semantic_memory_id(report)
                if report.outcome in SUCCESSFUL_CONSOLIDATION_OUTCOMES and semantic_id is not None:
                    semantic_ids_for_cluster.append(semantic_id)
                    created_memory_ids.append(semantic_id)
                    memories_created += 1
                    await self._create_consolidation_relations(
                        semantic_id,
                        source_ids,
                        dream_run_id,
                        seed.id,
                        dream,
                        proposal,
                    )
                cluster_report.dreamed_memories.append(
                    DreamedMemoryReport(
                        content=proposal["content"],
                        memory_type=proposal["memory_type"],
                        confidence=proposal["confidence"],
                        created_memory_id=semantic_id,
                        outcome=report.outcome,
                        source_memory_ids=_evidence_source_ids(proposal, source_ids),
                    )
                )

            if semantic_ids_for_cluster:
                await self.memory_store.mark_memories_consolidated_after_dream(
                    source_ids,
                    created_memory_ids=semantic_ids_for_cluster,
                    penalty=settings.dreaming_source_penalty,
                    min_importance=settings.dreaming_min_source_importance,
                    dream_run_id=dream_run_id,
                )

        completed_run = await self.dream_store.complete_run(
            dream_run_id,
            seed_memory_ids=seed_memory_ids,
            source_memory_ids=list(dict.fromkeys(source_memory_ids)),
            created_memory_ids=list(dict.fromkeys(created_memory_ids)),
            clusters_considered=clusters_considered,
            clusters_dreamed=clusters_dreamed,
            memories_created=memories_created,
            dry_run=dry_run,
        )
        return DreamRunReport(
            dream_run_id=completed_run.id,
            namespace=namespace,
            status=completed_run.status,
            started_at=started_at,
            completed_at=completed_run.completed_at,
            clusters_considered=clusters_considered,
            clusters_dreamed=clusters_dreamed,
            memories_created=memories_created,
            dry_run=dry_run,
            clusters=cluster_reports,
            error=completed_run.error,
        )

    async def _build_cluster(self, namespace: str, seed: Memory) -> list[Memory] | None:
        """Build and validate an active episodic cluster from one seed.

        Args:
            namespace: Memory partition guard.
            seed: Active episodic memory selected as the graph seed.

        Returns:
            Ordered active episodic memories or `None` if cluster thresholds fail.
        """
        node_refs = await self.graph_store.traverse_hebbian_cluster(
            seed.id,
            max_depth=settings.dreaming_max_depth,
            min_weight=settings.dreaming_min_edge_weight,
            max_nodes=settings.dreaming_cluster_max_size,
        )
        ordered_ids = [node_id for node_id, _weight, _depth in node_refs]
        memories = await self.memory_store.get_active_episodic_by_ids(
            ordered_ids,
            namespace=namespace,
        )
        memory_by_id = {memory.id: memory for memory in memories}
        cluster = [
            memory_by_id[memory_id] for memory_id in ordered_ids if memory_id in memory_by_id
        ]
        if len(cluster) < settings.dreaming_cluster_min_size:
            return None
        total_importance = sum(memory.base_importance for memory in cluster)
        if total_importance < settings.dreaming_min_cluster_total_importance:
            return None
        return cluster

    async def _create_consolidation_relations(
        self,
        semantic_id: UUID,
        source_ids: list[UUID],
        dream_run_id: UUID,
        seed_id: UUID,
        dream: dict[str, Any],
        proposal: dict[str, Any],
    ) -> None:
        """Link a semantic memory to every episodic source it consolidates.

        Args:
            semantic_id: Created or reinforced durable memory ID.
            source_ids: Episodic source IDs in cluster order.
            dream_run_id: Audit row ID.
            seed_id: Seed memory that initiated the cluster.
            dream: Raw dream response containing the cluster summary.
            proposal: Validated memory proposal containing evidence/rationale.
        """
        for source_id in source_ids:
            await self.relation_store.create_relation(
                source_id=semantic_id,
                target_id=source_id,
                relation_type="consolidates",
                confidence=settings.dreaming_relation_confidence,
                rationale=proposal.get("rationale"),
                metadata={
                    "dream_run_id": str(dream_run_id),
                    "evidence_indices": proposal["evidence_indices"],
                    "cluster_seed_id": str(seed_id),
                    "dream_summary": dream.get("summary"),
                },
            )


def _is_duplicate_in_current_run(cluster_ids: set[UUID], processed: list[set[UUID]]) -> bool:
    """Detect near-identical source sets within one dreaming run.

    Args:
        cluster_ids: Candidate cluster source IDs.
        processed: Previously accepted cluster source sets.

    Returns:
        Whether overlap is high enough to skip the candidate.
    """
    if not cluster_ids:
        return True
    for existing in processed:
        overlap = len(cluster_ids & existing) / min(len(cluster_ids), len(existing))
        if overlap >= 0.8:
            return True
    return False


def _validated_proposals(raw_memories: object, source_count: int) -> list[dict[str, Any]]:
    """Filter LLM memory proposals to safe, supported candidates.

    Args:
        raw_memories: Untrusted `memories` value from the provider.
        source_count: Number of memories available for evidence indices.

    Returns:
        Confidence-sorted validated proposals capped by settings.
    """
    if not isinstance(raw_memories, list):
        return []

    proposals: list[dict[str, Any]] = []
    for item in raw_memories:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        memory_type = str(item.get("memory_type") or "semantic")
        confidence = _float_or_none(item.get("confidence"))
        evidence_indices = item.get("evidence_indices")
        if not content or memory_type not in ALLOWED_DREAM_MEMORY_TYPES:
            continue
        if confidence is None or confidence < settings.dreaming_min_synthesis_confidence:
            continue
        if not isinstance(evidence_indices, list) or not evidence_indices:
            continue
        valid_indices = []
        for index in evidence_indices:
            if isinstance(index, int) and 0 <= index < source_count:
                valid_indices.append(index)
        if not valid_indices:
            continue
        proposals.append(
            {
                "content": content,
                "memory_type": memory_type,
                "confidence": max(0.0, min(confidence, 1.0)),
                "evidence_indices": list(dict.fromkeys(valid_indices)),
                "rationale": item.get("rationale"),
            }
        )

    proposals.sort(key=lambda item: item["confidence"], reverse=True)
    return proposals[: settings.dreaming_max_memories_per_cluster]


def _float_or_none(value: object) -> float | None:
    """Convert provider confidence values defensively.

    Args:
        value: Untrusted provider value.

    Returns:
        Parsed float or `None` when conversion fails.
    """
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _semantic_memory_id(report: Any) -> UUID | None:
    """Extract the semantic memory ID affected by reconciliation.

    Args:
        report: MemorySubmissionReport-like object.

    Returns:
        Created ID, or first affected ID for reinforcement outcomes.
    """
    if report.created_memory_id is not None:
        return report.created_memory_id
    if report.affected_memory_ids:
        return report.affected_memory_ids[0]
    return None


def _evidence_source_ids(proposal: dict[str, Any], source_ids: list[UUID]) -> list[UUID]:
    """Map evidence indices from a proposal back to episodic source IDs.

    Args:
        proposal: Validated provider proposal.
        source_ids: Cluster source IDs in evidence order.

    Returns:
        Source IDs supporting the proposed memory.
    """
    return [source_ids[index] for index in proposal["evidence_indices"]]
