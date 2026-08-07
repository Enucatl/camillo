from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from camillo.db.session import get_db
from camillo.schemas.memory import MemoryStatsResponse
from camillo.stores.memory_store import MemoryStore

router = APIRouter()


@router.post("/memory_stats", response_model=MemoryStatsResponse)
async def memory_stats(
    workspace: str | None = None, db: AsyncSession = Depends(get_db)
) -> MemoryStatsResponse:
    """Return corpus counts with optional workspace diagnostics."""
    stats = await MemoryStore(db).memory_stats(workspace)
    await db.commit()
    return MemoryStatsResponse(workspace=workspace, **stats)
