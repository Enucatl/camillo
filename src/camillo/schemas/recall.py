from uuid import UUID

from pydantic import BaseModel


class RecallRequest(BaseModel):
    namespace: str
    query: str
    top_k: int | None = None


class RecalledMemory(BaseModel):
    id: UUID
    namespace: str
    raw_content: str
    type: str
    base_importance: float
    access_count: int
    score: float


class RecallResponse(BaseModel):
    query: str
    namespace: str
    memories: list[RecalledMemory]
