from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.ingestion_service import IngestionService
from camillo.db.session import get_db
from camillo.schemas.ingest import IngestRequest, IngestResponse
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, db: AsyncSession = Depends(get_db)) -> IngestResponse:
    """Capture an automatic redacted conversation episode."""
    memory = await IngestionService(MemoryStore(db), LiteLLMService()).ingest_interaction(
        request.user_msg, request.ai_msg, request.session_id, request.workspace
    )
    await db.commit()
    return IngestResponse(
        memory_id=memory.id,
        workspace=memory.workspace,
        type=memory.type,
        base_importance=memory.base_importance,
    )
