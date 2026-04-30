from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.ingestion_service import IngestionService
from camillo.db.session import get_db
from camillo.schemas.ingest import IngestRequest, IngestResponse
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, db: AsyncSession = Depends(get_db)) -> IngestResponse:
    memory_store = MemoryStore(db)
    graph_store = GraphStore(db)
    llm_service = LiteLLMService()
    service = IngestionService(memory_store, graph_store, llm_service)

    memory = await service.ingest_interaction(
        namespace=request.namespace,
        user_msg=request.user_msg,
        ai_msg=request.ai_msg,
        session_id=request.session_id,
    )
    await db.commit()

    return IngestResponse(
        memory_id=memory.id,
        namespace=memory.namespace,
        type=memory.type,
        base_importance=memory.base_importance,
    )
