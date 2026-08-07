from uuid import UUID

from pydantic import BaseModel


class IngestRequest(BaseModel):
    """Automatic hook capture payload."""

    user_msg: str
    ai_msg: str
    session_id: str | None = None
    workspace: str | None = None


class IngestResponse(BaseModel):
    """Identifier and classification of the captured episode."""

    memory_id: UUID
    workspace: str | None
    type: str
    base_importance: float
