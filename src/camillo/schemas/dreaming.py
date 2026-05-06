from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DreamRequest(BaseModel):
    """Manual request to run dreaming for one namespace.

    The request is intentionally small because scheduling policy belongs to the
    worker while the endpoint is only an operational trigger.
    """

    namespace: str
    seed_limit: int | None = None
    dry_run: bool | None = None


class DreamedMemoryReport(BaseModel):
    """Serializable result for one synthesized durable memory candidate.

    Reports include rejected and dry-run candidates so operators can inspect
    LLM synthesis without inferring behavior from database writes.
    """

    content: str
    memory_type: str = "semantic"
    confidence: float = Field(ge=0.0, le=1.0)
    created_memory_id: UUID | None = None
    outcome: str
    source_memory_ids: list[UUID]


class DreamClusterReport(BaseModel):
    """Serializable result for one graph-connected episodic cluster.

    The cluster report preserves source IDs so consolidation can be audited
    without exposing raw episodic content in the response.
    """

    seed_memory_id: UUID
    source_memory_ids: list[UUID]
    summary: str | None = None
    dreamed_memories: list[DreamedMemoryReport] = Field(default_factory=list)


class DreamRunReport(BaseModel):
    """Top-level report for a dreaming run.

    The response mirrors the `dream_runs` counters while adding per-cluster
    details for dry-run review and endpoint diagnostics.
    """

    dream_run_id: UUID
    namespace: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    clusters_considered: int
    clusters_dreamed: int
    memories_created: int
    dry_run: bool
    clusters: list[DreamClusterReport] = Field(default_factory=list)
    error: str | None = None
