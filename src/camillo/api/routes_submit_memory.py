from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import get_inference_service
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import get_db
from camillo.schemas.submit_memory import (
    ForgetMemoryRequest,
    MemorySubmissionReport,
    RememberMemoryRequest,
    ReplaceMemoryRequest,
)
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


def _service(db: AsyncSession) -> MemoryReconciliationService:
    """Build the explicit durable-memory operation boundary."""
    store = MemoryStore(db)
    provider = get_inference_service()
    return MemoryReconciliationService(store, RecallService(store, provider), provider)


@router.post("/remember_memory", response_model=MemorySubmissionReport)
async def remember_memory(
    request: RememberMemoryRequest, db: AsyncSession = Depends(get_db)
) -> MemorySubmissionReport:
    """Create or reinforce a durable memory."""
    report = await _service(db).remember_memory(
        request.content, request.memory_type, request.evidence, request.workspace
    )
    await db.commit()
    return report


@router.post("/replace_memory", response_model=MemorySubmissionReport)
async def replace_memory(
    request: ReplaceMemoryRequest, db: AsyncSession = Depends(get_db)
) -> MemorySubmissionReport:
    """Replace one explicit active memory."""
    report = await _service(db).replace_memory(
        request.memory_id, request.content, request.memory_type, request.evidence
    )
    await db.commit()
    return report


@router.post("/forget_memory", response_model=MemorySubmissionReport)
async def forget_memory(
    request: ForgetMemoryRequest, db: AsyncSession = Depends(get_db)
) -> MemorySubmissionReport:
    """Forget one explicit active memory."""
    report = await _service(db).forget_memory(request.memory_id, request.reason)
    await db.commit()
    return report
