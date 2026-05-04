from typing import Protocol

from camillo.cognitive.recall_service import RecallService
from camillo.db.models import Memory
from camillo.interfaces import EmbeddingProvider, MemoryStoreProtocol
from camillo.schemas.submit_memory import (
    DurableMemoryType,
    MemoryRelationReport,
    MemoryRelationshipClassification,
    MemorySubmissionReport,
)
from camillo.settings import settings
from camillo.stores.relation_store import RelationStore

MEANINGFUL_RELATIONS = {
    "confirms",
    "extends",
    "contradicts",
    "supersedes",
    "forgets",
    "duplicate",
}


class RelationshipClassifier(Protocol):
    """Provider behavior needed by memory reconciliation."""

    async def classify_memory_relationships(
        self,
        intent: str,
        new_content: str,
        existing_memories: list[Memory],
    ) -> list[MemoryRelationshipClassification]:
        """Classify the candidate against related memories."""


class MemoryReconciliationService:
    """Own durable memory policy and lifecycle decisions.

    MCP and HTTP callers expose simple cognitive actions while this service
    handles duplicate reinforcement, contradiction context, and status changes.
    """

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        relation_store: RelationStore,
        recall_service: RecallService,
        llm_service: EmbeddingProvider | RelationshipClassifier,
    ):
        """Wire reconciliation to retrieval, storage, and classification.

        Args:
            memory_store: Memory persistence boundary.
            relation_store: Semantic relation persistence boundary.
            recall_service: Existing Phase 2 recall pipeline.
            llm_service: Provider for embeddings and relationship classification.
        """
        self.memory_store = memory_store
        self.relation_store = relation_store
        self.recall_service = recall_service
        self.llm_service = llm_service

    async def submit_memory(
        self,
        namespace: str,
        content: str,
        intent: str = "auto",
        memory_type: str | None = None,
        evidence: str | None = None,
        confidence: float | None = None,
    ) -> MemorySubmissionReport:
        """Reconcile a durable memory candidate with related active memories.

        Args:
            namespace: Logical partition for the memory.
            content: Candidate memory content.
            intent: Caller intent: auto, remember, correct, or forget.
            memory_type: Durable memory type, defaulting to semantic.
            evidence: Optional support text stored as metadata.
            confidence: Optional caller confidence.

        Returns:
            Transparent report of created, reinforced, or lifecycle actions.
        """
        normalized_content = content.strip()
        if not namespace.strip() or not normalized_content:
            return MemorySubmissionReport(
                outcome="ignored_low_confidence",
                message="Namespace and content are required.",
            )

        durable_type = self._normalize_memory_type(memory_type)
        candidate_confidence = 0.8 if confidence is None else max(0.0, min(confidence, 1.0))
        if candidate_confidence <= 0.0 and intent != "forget":
            return MemorySubmissionReport(
                outcome="ignored_low_confidence",
                message="Memory confidence was too low to store.",
            )

        related = await self.recall_service.recall(
            namespace=namespace,
            query=normalized_content,
            top_k=settings.recall_top_k,
            include_hebbian=False,
        )
        related_memories = [candidate.memory for candidate in related]

        if intent == "forget":
            return await self._forget_related(related_memories, normalized_content)

        try:
            classifications = await self.llm_service.classify_memory_relationships(
                intent,
                normalized_content,
                related_memories,
            )
        except Exception:
            classifications = [
                MemoryRelationshipClassification(
                    index=index,
                    relation="unrelated",
                    confidence=0.0,
                    rationale="classification unavailable",
                )
                for index, _memory in enumerate(related_memories)
            ]
        best = self._best_classification(classifications, len(related_memories))
        if best is None or best.relation not in MEANINGFUL_RELATIONS:
            memory = await self._create_memory(
                namespace,
                normalized_content,
                durable_type,
                candidate_confidence,
                evidence,
                intent,
                best,
            )
            return MemorySubmissionReport(
                outcome="created",
                created_memory_id=memory.id,
                message="Stored new memory; no related active memory required lifecycle changes.",
            )

        old_memory = related_memories[best.index]
        if best.relation in {"duplicate", "confirms"} and best.confidence >= 0.8:
            await self.memory_store.reinforce_memory(old_memory.id)
            return MemorySubmissionReport(
                outcome="ignored_duplicate",
                affected_memory_ids=[old_memory.id],
                message="Reinforced an existing related memory instead of creating a duplicate.",
            )

        if best.relation == "forgets" and best.confidence >= 0.7:
            await self.memory_store.update_memory_status(
                old_memory.id,
                "deprecated",
                reason=best.rationale,
            )
            return MemorySubmissionReport(
                outcome="deprecated_old_memory",
                affected_memory_ids=[old_memory.id],
                message="Deprecated the related memory.",
            )

        memory = await self._create_memory(
            namespace,
            normalized_content,
            durable_type,
            candidate_confidence
            if best.resolution != "needs_review"
            else min(candidate_confidence, 0.5),
            evidence,
            intent,
            best,
        )

        relation_type = self._relation_type(best)
        relations: list[MemoryRelationReport] = []
        affected: list = []
        if relation_type is not None:
            relation = await self.relation_store.create_relation(
                memory.id,
                old_memory.id,
                relation_type,
                confidence=best.confidence,
                rationale=best.rationale,
                metadata={
                    "old_memory_refinement": best.old_memory_refinement,
                    "new_memory_refinement": best.new_memory_refinement,
                    "contradiction_type": best.contradiction_type,
                    "resolution": best.resolution,
                },
            )
            relations.append(
                MemoryRelationReport(
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    rationale=relation.rationale,
                )
            )

        outcome = "created"
        message = "Stored new memory."
        if self._should_supersede(best):
            await self.memory_store.update_memory_status(
                old_memory.id,
                "superseded",
                reason=best.rationale,
                superseded_by=memory.id,
            )
            affected.append(old_memory.id)
            outcome = "superseded_old_memory"
            message = "Stored new memory and superseded one older memory."
        elif self._should_deprecate(best):
            await self.memory_store.update_memory_status(
                old_memory.id,
                "deprecated",
                reason=best.rationale,
            )
            affected.append(old_memory.id)
            outcome = "deprecated_old_memory"
            message = "Stored new memory and deprecated one older memory."

        return MemorySubmissionReport(
            outcome=outcome,
            created_memory_id=memory.id,
            affected_memory_ids=affected,
            relations=relations,
            message=message,
        )

    async def _forget_related(
        self,
        related_memories: list[Memory],
        reason: str,
    ) -> MemorySubmissionReport:
        """Deprecate the best recalled memories for explicit forget requests."""
        if not related_memories:
            return MemorySubmissionReport(
                outcome="no_related_memory_found",
                message="No related active memory was found to forget.",
            )

        affected = []
        for memory in related_memories:
            await self.memory_store.update_memory_status(
                memory.id,
                "deprecated",
                reason=reason,
            )
            affected.append(memory.id)
        return MemorySubmissionReport(
            outcome="deprecated_old_memory",
            affected_memory_ids=affected,
            message="Deprecated related active memories.",
        )

    async def _create_memory(
        self,
        namespace: str,
        content: str,
        memory_type: DurableMemoryType,
        confidence: float,
        evidence: str | None,
        intent: str,
        classification: MemoryRelationshipClassification | None,
    ) -> Memory:
        """Persist a new durable memory with reconciliation metadata."""
        embedding = await self.llm_service.get_embedding(content)
        return await self.memory_store.insert_memory(
            namespace=namespace,
            raw_content=content,
            embedding=embedding,
            memory_type=memory_type,
            base_importance=confidence,
            confidence=confidence,
            source="mcp_submit_memory",
            metadata={
                "evidence": evidence,
                "intent": intent,
                "creation_mode": "submit_memory",
                "contradiction_type": classification.contradiction_type
                if classification is not None
                else "none",
                "resolution": classification.resolution
                if classification is not None
                else "keep_both",
            },
        )

    def _best_classification(
        self,
        classifications: list[MemoryRelationshipClassification],
        related_count: int,
    ) -> MemoryRelationshipClassification | None:
        """Pick the highest-confidence meaningful classification."""
        meaningful = [
            classification
            for classification in classifications
            if classification.relation in MEANINGFUL_RELATIONS
            and 0 <= classification.index < related_count
        ]
        if not meaningful:
            return None
        return max(meaningful, key=lambda classification: classification.confidence)

    def _normalize_memory_type(self, memory_type: str | None) -> DurableMemoryType:
        """Default and validate the durable memory type."""
        allowed = {"semantic", "preference", "procedural", "relationship", "profile", "core"}
        if memory_type is None:
            return "semantic"
        if memory_type not in allowed:
            return "semantic"
        return memory_type  # type: ignore[return-value]

    def _relation_type(self, classification: MemoryRelationshipClassification) -> str | None:
        """Map classifier output to persisted relation labels."""
        if classification.relation == "extends" and classification.confidence >= 0.7:
            return "extends"
        if classification.relation == "supersedes" and classification.confidence >= 0.7:
            return "supersedes"
        if classification.relation == "contradicts":
            if classification.resolution == "refine_old":
                return "refines"
            if classification.resolution == "create_exception":
                return "exception_to"
            if classification.resolution == "supersede_old":
                return "supersedes"
            return "contradicts"
        return None

    def _should_supersede(self, classification: MemoryRelationshipClassification) -> bool:
        """Decide whether policy allows replacing an older memory."""
        return (classification.relation == "supersedes" and classification.confidence >= 0.7) or (
            classification.relation == "contradicts"
            and classification.resolution == "supersede_old"
        )

    def _should_deprecate(self, classification: MemoryRelationshipClassification) -> bool:
        """Decide whether policy should deprecate an older memory."""
        return (
            classification.relation == "contradicts"
            and classification.resolution == "deprecate_old"
        )
