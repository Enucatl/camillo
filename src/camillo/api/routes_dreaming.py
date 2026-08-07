from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import get_inference_service
from camillo.cognitive.dreaming_service import DreamingService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import get_db
from camillo.schemas.dreaming import DreamRequest, DreamRunReport
from camillo.stores.dream_store import DreamStore
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/dream", response_model=DreamRunReport)
async def dream(request: DreamRequest, db: AsyncSession = Depends(get_db)) -> DreamRunReport:
    """Run one operational dreaming pass."""
    store = MemoryStore(db)
    provider = get_inference_service()
    service = DreamingService(
        store,
        DreamStore(db),
        MemoryReconciliationService(store, RecallService(store, provider), provider),
        provider,
    )
    report = await service.run_once(seed_limit=request.seed_limit, dry_run=request.dry_run)
    await db.commit()
    return report
