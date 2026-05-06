from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import get_db
from camillo.schemas.submit_memory import MemorySubmissionReport, SubmitMemoryRequest
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore

router = APIRouter()


@router.post("/submit_memory", response_model=MemorySubmissionReport)
async def submit_memory(
    request: SubmitMemoryRequest,
    db: AsyncSession = Depends(get_db),
) -> MemorySubmissionReport:
    """Submit durable memory through the reconciliation layer."""
    memory_store = MemoryStore(db)
    graph_store = GraphStore(db)
    relation_store = RelationStore(db)
    llm_service = LiteLLMService()
    recall_service = RecallService(memory_store, graph_store, llm_service)
    service = MemoryReconciliationService(
        memory_store,
        relation_store,
        recall_service,
        llm_service,
    )

    report = await service.submit_memory(
        namespace=request.namespace,
        content=request.content,
        intent=request.intent,
        memory_type=request.memory_type,
        scope=request.scope,
        evidence=request.evidence,
        confidence=request.confidence,
    )
    await db.commit()
    return report
