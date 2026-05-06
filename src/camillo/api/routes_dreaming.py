from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.dreaming_service import DreamingService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import get_db
from camillo.schemas.dreaming import DreamRequest, DreamRunReport
from camillo.stores.dream_store import DreamStore
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore

router = APIRouter()


@router.post("/dream", response_model=DreamRunReport)
async def dream(
    request: DreamRequest,
    db: AsyncSession = Depends(get_db),
) -> DreamRunReport:
    """Run one admin-triggered dreaming pass for a namespace."""
    memory_store = MemoryStore(db)
    graph_store = GraphStore(db)
    relation_store = RelationStore(db)
    dream_store = DreamStore(db)
    llm_service = LiteLLMService()
    recall_service = RecallService(memory_store, graph_store, llm_service)
    reconciliation_service = MemoryReconciliationService(
        memory_store,
        relation_store,
        recall_service,
        llm_service,
    )
    service = DreamingService(
        memory_store,
        graph_store,
        relation_store,
        dream_store,
        reconciliation_service,
        llm_service,
    )
    try:
        report = await service.run_once(
            namespace=request.namespace,
            seed_limit=request.seed_limit,
            dry_run=request.dry_run,
        )
        await db.commit()
        return report
    except Exception:
        await db.rollback()
        raise
