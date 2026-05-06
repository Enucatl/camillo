from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from camillo.schemas.recall import ScoreBreakdown


class MemoryRead(BaseModel):
    """General serialized memory representation."""

    id: UUID
    namespace: str
    session_id: str | None
    scope: str
    raw_content: str
    type: str
    status: str
    base_importance: float
    access_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata_json: dict[str, Any]


class MemoryStatsResponse(BaseModel):
    """Operational counts for one memory namespace.

    This shape is used by MCP clients that need to inspect whether a namespace
    has stored memories before deciding to recall, ingest, or reconcile data.
    """

    namespace: str
    total: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    by_scope: dict[str, int] = Field(default_factory=dict)


class McpRecalledMemory(BaseModel):
    """MCP-facing recalled memory without internal access bookkeeping."""

    id: UUID
    namespace: str
    scope: str
    raw_content: str
    type: str
    base_importance: float
    score: float
    score_breakdown: ScoreBreakdown
    source: str = "primary"
    linked_from: UUID | None = None
    edge_weight: float | None = None


class McpRecallResponse(BaseModel):
    """MCP recall result that treats adaptive recall bookkeeping as internal."""

    query: str
    namespace: str
    memories: list[McpRecalledMemory]
