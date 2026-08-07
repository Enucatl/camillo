from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import get_inference_service
from camillo.cognitive.recall_service import RecallService
from camillo.db.session import get_db
from camillo.schemas.recall import RecalledMemory, RecallRequest, RecallResponse, ScoreBreakdown
from camillo.settings import settings
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest, db: AsyncSession = Depends(get_db)) -> RecallResponse:
    """Return active corpus memories and reinforce public recall results."""
    service = RecallService(MemoryStore(db), get_inference_service())
    candidates = await service.recall(
        request.query, request.top_k or settings.recall_top_k, request.workspace
    )
    await db.commit()
    return RecallResponse(
        query=request.query,
        workspace=request.workspace,
        memories=[
            RecalledMemory(
                id=c.memory.id,
                workspace=c.memory.workspace,
                raw_content=c.memory.raw_content,
                type=c.memory.type,
                base_importance=c.memory.base_importance,
                access_count=c.memory.access_count,
                score=c.final_score or 0.0,
                score_breakdown=ScoreBreakdown(
                    retrieval_score=c.retrieval_score,
                    rerank_score=c.rerank_score,
                    activation_score=c.activation_score or 0.0,
                    workspace_affinity_score=c.workspace_affinity_score or 0.0,
                    final_score=c.final_score or 0.0,
                    vector_score=c.vector_score,
                    text_score=c.text_score,
                    rrf_score=c.rrf_score,
                ),
            )
            for c in candidates
        ],
    )
