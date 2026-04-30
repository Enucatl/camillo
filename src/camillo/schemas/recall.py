from uuid import UUID

from pydantic import BaseModel


class RecallRequest(BaseModel):
    """Request body for retrieving relevant memories."""

    namespace: str
    query: str
    top_k: int | None = None


class RecalledMemory(BaseModel):
    """Memory payload returned by recall."""

    id: UUID
    namespace: str
    raw_content: str
    type: str
    base_importance: float
    access_count: int
    score: float


class RecallResponse(BaseModel):
    """Response body for a recall query."""

    query: str
    namespace: str
    memories: list[RecalledMemory]
