from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DreamRequest(BaseModel):
    """Manual trigger for one corpus-wide dreaming pass."""

    seed_limit: int | None = None
    dry_run: bool | None = None


class DreamRunReport(BaseModel):
    """Operational audit result for one one-shot run."""

    dream_run_id: UUID
    status: str
    started_at: datetime
    completed_at: datetime | None
    source_memory_ids: list[UUID]
    created_memory_ids: list[UUID]
    memories_created: int
    dry_run: bool
