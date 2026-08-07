from uuid import UUID

from camillo.ai.llm_service import InferenceService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.schemas.dreaming import DreamRunReport
from camillo.settings import settings
from camillo.stores.dream_store import DreamStore
from camillo.stores.memory_store import MemoryStore


class DreamingService:
    """Run one corpus-wide, one-proposal consolidation pass."""

    def __init__(
        self,
        memory_store: MemoryStore,
        dream_store: DreamStore,
        reconciliation_service: MemoryReconciliationService,
        llm_service: InferenceService,
    ):
        """Wire storage, audit, synthesis, and normal deduplication policy."""
        self.memory_store = memory_store
        self.dream_store = dream_store
        self.reconciliation_service = reconciliation_service
        self.llm_service = llm_service

    async def run_once(
        self, *, seed_limit: int | None = None, dry_run: bool | None = None
    ) -> DreamRunReport:
        """Consolidate one qualifying batch and mark sources only on success."""
        effective_dry_run = settings.dreaming_dry_run if dry_run is None else dry_run
        seeds = await self.memory_store.select_dream_seeds(
            limit=seed_limit or settings.dreaming_seed_limit,
            min_activation=0.0,
            decay_rate=settings.decay_rate,
        )
        run = await self.dream_store.create_run(
            dry_run=effective_dry_run, metadata={"seed_limit": len(seeds)}
        )
        batch = await self._qualifying_batch(seeds)
        if len(batch) < 2:
            completed = await self.dream_store.complete_run(
                run.id,
                seed_memory_ids=[s.id for s in seeds],
                source_memory_ids=[],
                created_memory_ids=[],
                clusters_considered=0,
                clusters_dreamed=0,
                memories_created=0,
                dry_run=effective_dry_run,
            )
            return _report(completed, effective_dry_run)
        proposal = await self.llm_service.synthesize_dream([seed.raw_content for seed in batch])
        content = str(proposal.get("content") or "").strip()
        confidence = float(proposal.get("confidence") or 0.0)
        created_ids: list[UUID] = []
        dreamed = 0
        if (
            content
            and confidence >= settings.dreaming_min_synthesis_confidence
            and not effective_dry_run
        ):
            result = await self.reconciliation_service.remember_memory(
                content, str(proposal.get("memory_type") or "fact"), evidence="dream synthesis"
            )
            if result.memory_id is not None and result.outcome in {"created", "reinforced"}:
                created_ids.append(result.memory_id)
                dreamed = 1
                await self.memory_store.mark_memories_consolidated_after_dream(
                    [s.id for s in batch], created_memory_ids=created_ids, dream_run_id=run.id
                )
        completed = await self.dream_store.complete_run(
            run.id,
            seed_memory_ids=[s.id for s in seeds],
            source_memory_ids=[s.id for s in batch],
            created_memory_ids=created_ids,
            clusters_considered=1,
            clusters_dreamed=dreamed,
            memories_created=len(created_ids),
            dry_run=effective_dry_run,
        )
        return _report(completed, effective_dry_run)

    async def _qualifying_batch(self, seeds: list) -> list:
        """Choose one semantically coherent episode batch without graph traversal."""
        for seed in seeds:
            matches = await self.memory_store.vector_candidates(
                seed.embedding, settings.dreaming_batch_size
            )
            batch = [
                memory
                for memory, similarity in matches
                if memory.type == "episode"
                and memory.status == "active"
                and similarity >= settings.dreaming_min_similarity
            ]
            if seed not in batch:
                batch.insert(0, seed)
            if len(batch) >= 2:
                return batch[: settings.dreaming_batch_size]
        return []


def _report(run: object, dry_run: bool) -> DreamRunReport:
    """Convert an audit model into the small operational response."""
    return DreamRunReport(
        dream_run_id=run.id,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        source_memory_ids=list(run.source_memory_ids or []),
        created_memory_ids=list(run.created_memory_ids or []),
        memories_created=run.memories_created,
        dry_run=dry_run,
    )
