import re
from uuid import UUID

from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.redaction import redact_secrets
from camillo.interfaces import EmbeddingProvider, MemoryStoreProtocol
from camillo.schemas.submit_memory import MemorySubmissionReport


def _normalize(content: str) -> str:
    """Normalize whitespace and case for deterministic duplicate checks."""
    return re.sub(r"\s+", " ", content).strip().casefold()


class MemoryReconciliationService:
    """Own safe durable creation, replacement, and forgetting boundaries."""

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        recall_service: RecallService,
        llm_service: EmbeddingProvider,
    ):
        """Wire read-only candidate search to storage and embeddings."""
        self.memory_store = memory_store
        self.recall_service = recall_service
        self.llm_service = llm_service

    async def remember_memory(
        self,
        content: str,
        memory_type: str = "fact",
        evidence: str | None = None,
        workspace: str | None = None,
    ) -> MemorySubmissionReport:
        """Create or reinforce one durable memory after safe duplicate checks."""
        content = redact_secrets(content).strip()
        if not content:
            return MemorySubmissionReport(
                outcome="rejected", message="Content is empty after redaction."
            )
        embedding = await self.llm_service.get_embedding(content, domain="durable_memory_embedding")
        candidates = await self.recall_service.search(content, top_k=10, workspace=workspace)
        normalized = _normalize(content)
        exact = next(
            (c.memory for c in candidates if _normalize(c.memory.raw_content) == normalized), None
        )
        semantic = next(
            (
                c.memory
                for c in candidates
                if c.memory.type in {"fact", "preference", "procedure"}
                and c.retrieval_score >= 0.95
            ),
            None,
        )
        duplicate = exact or semantic
        if duplicate is not None:
            reinforced = await self.memory_store.reinforce_memory(duplicate.id)
            if reinforced is None:
                return MemorySubmissionReport(
                    outcome="inactive",
                    memory_id=duplicate.id,
                    message="Duplicate target is inactive.",
                )
            return MemorySubmissionReport(
                outcome="reinforced", memory_id=duplicate.id, message="Reinforced existing memory."
            )
        memory = await self.memory_store.insert_memory(
            raw_content=content,
            embedding=embedding,
            memory_type=memory_type,
            base_importance=0.8,
            workspace=workspace,
            metadata={"evidence": evidence} if evidence else None,
        )
        return MemorySubmissionReport(
            outcome="created", memory_id=memory.id, message="Created durable memory."
        )

    async def replace_memory(
        self, memory_id: UUID, content: str, memory_type: str = "fact", evidence: str | None = None
    ) -> MemorySubmissionReport:
        """Replace exactly one active target, preserving explicit lineage."""
        targets = await self.memory_store.get_memories_by_ids([memory_id])
        if not targets:
            existing = await self.memory_store.get_memories_by_ids([memory_id], active_only=False)
            return MemorySubmissionReport(
                outcome="inactive" if existing else "not_found",
                memory_id=memory_id,
                message="Target is not active." if existing else "Target not found.",
            )
        redacted = redact_secrets(content).strip()
        if not redacted:
            return MemorySubmissionReport(
                outcome="rejected", message="Content is empty after redaction."
            )
        embedding = await self.llm_service.get_embedding(
            redacted, domain="durable_memory_embedding"
        )
        replacement = await self.memory_store.insert_memory(
            raw_content=redacted,
            embedding=embedding,
            memory_type=memory_type,
            base_importance=0.8,
            workspace=targets[0].workspace,
            metadata={"evidence": evidence} if evidence else None,
        )
        await self.memory_store.update_memory_status(
            memory_id, "superseded", reason="Explicit replacement", superseded_by=replacement.id
        )
        return MemorySubmissionReport(
            outcome="replaced", memory_id=replacement.id, message="Replaced target memory."
        )

    async def forget_memory(
        self, memory_id: UUID, reason: str | None = None
    ) -> MemorySubmissionReport:
        """Forget exactly one active target and never mutate search results in bulk."""
        targets = await self.memory_store.get_memories_by_ids([memory_id])
        if not targets:
            existing = await self.memory_store.get_memories_by_ids([memory_id], active_only=False)
            return MemorySubmissionReport(
                outcome="inactive" if existing else "not_found",
                memory_id=memory_id,
                message="Target is not active." if existing else "Target not found.",
            )
        await self.memory_store.update_memory_status(
            memory_id, "deprecated", reason=reason or "Explicitly forgotten"
        )
        return MemorySubmissionReport(
            outcome="forgotten", memory_id=memory_id, message="Forgotten target memory."
        )
