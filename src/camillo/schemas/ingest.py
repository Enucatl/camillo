from uuid import UUID

from pydantic import BaseModel


class IngestRequest(BaseModel):
    """Request body for storing a conversation turn."""

    namespace: str
    user_msg: str
    ai_msg: str
    session_id: str | None = None


class IngestResponse(BaseModel):
    """Response body for a stored memory."""

    memory_id: UUID
    namespace: str
    type: str
    base_importance: float
