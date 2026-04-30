from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cognitive_memory.ai.llm_service import LiteLLMService
from cognitive_memory.cognitive.recall_service import RecallService
from cognitive_memory.db.session import get_db
from cognitive_memory.schemas.recall import RecallRequest, RecallResponse, RecalledMemory
from cognitive_memory.settings import settings
from cognitive_memory.stores.graph_store import GraphStore
from cognitive_memory.stores.memory_store import MemoryStore

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
