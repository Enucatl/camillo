from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class MemoryRead(BaseModel):
    """General serialized memory representation."""

    id: UUID
    namespace: str
    session_id: str | None
    raw_content: str
    type: str
    status: str
    base_importance: float
    access_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata_json: dict[str, Any]
