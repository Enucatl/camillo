from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.recall_service import RecallService
from camillo.db.session import get_db
from camillo.schemas.recall import RecallRequest, RecallResponse, RecalledMemory
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest, db: AsyncSession = Depends(get_db)) -> RecallResponse:
    memory_store = MemoryStore(db)
    graph_store = GraphStore(db)
    llm_service = LiteLLMService()
    service = RecallService(memory_store, graph_store, llm_service)

    memories = await service.recall(
        namespace=request.namespace,
        query=request.query,
        top_k=request.top_k or settings.recall_top_k,
    )
    await db.commit()

    return RecallResponse(
        query=request.query,
        namespace=request.namespace,
        memories=[RecalledMemory(**memory) for memory in memories],
    )
