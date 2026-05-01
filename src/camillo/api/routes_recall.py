from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.recall_service import RecallService
from camillo.db.session import get_db
from camillo.schemas.recall import RecalledMemory, RecallRequest, RecallResponse, ScoreBreakdown
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest, db: AsyncSession = Depends(get_db)) -> RecallResponse:
    """Own API serialization so the service can return rich internal candidates.

    Args:
        request: Recall input and optional graph-expansion preference.
        db: Request-scoped database session.

    Returns:
        A response that preserves the old score field and adds score provenance.
    """
    memory_store = MemoryStore(db)
    graph_store = GraphStore(db)
    llm_service = LiteLLMService()
    service = RecallService(memory_store, graph_store, llm_service)

    candidates = await service.recall(
        namespace=request.namespace,
        query=request.query,
        top_k=request.top_k or settings.recall_top_k,
        include_hebbian=request.include_hebbian,
    )
    await db.commit()

    return RecallResponse(
        query=request.query,
        namespace=request.namespace,
        memories=[
            RecalledMemory(
                id=candidate.memory.id,
                namespace=candidate.memory.namespace,
                raw_content=candidate.memory.raw_content,
                type=candidate.memory.type,
                base_importance=candidate.memory.base_importance,
                access_count=candidate.memory.access_count,
                score=candidate.final_score or 0.0,
                source=candidate.source,
                linked_from=candidate.linked_from,
                edge_weight=candidate.edge_weight,
                score_breakdown=ScoreBreakdown(
                    retrieval_score=candidate.retrieval_score,
                    rerank_score=candidate.rerank_score,
                    activation_score=candidate.activation_score or 0.0,
                    final_score=candidate.final_score or 0.0,
                    vector_score=candidate.vector_score,
                    text_score=candidate.text_score,
                    rrf_score=candidate.rrf_score,
                ),
            )
            for candidate in candidates
        ],
    )
