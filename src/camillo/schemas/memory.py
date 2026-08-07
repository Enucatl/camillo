from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class MemoryRead(BaseModel):
    """Serialized memory row."""

    id: UUID
    workspace: str | None
    session_id: str | None
    raw_content: str
    type: str
    status: str
    base_importance: float
    access_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata_json: dict[str, Any]


class MemoryStatsResponse(BaseModel):
    """Corpus counts, optionally filtered to a workspace."""

    workspace: str | None
    total: int
    by_type: dict[str, int]
    by_status: dict[str, int]
