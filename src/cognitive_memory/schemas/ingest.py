from uuid import UUID

from pydantic import BaseModel


class IngestRequest(BaseModel):
    namespace: str
    user_msg: str
    ai_msg: str
    session_id: str | None = None


class IngestResponse(BaseModel):
    memory_id: UUID
    namespace: str
    type: str
    base_importance: float
