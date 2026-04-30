from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return a minimal liveness response for orchestration checks."""
    return {"status": "ok"}
