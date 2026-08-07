from typing import Literal
from uuid import UUID

from pydantic import BaseModel

MemoryType = Literal["episode", "fact", "preference", "procedure"]


class RememberMemoryRequest(BaseModel):
    """Input for a new durable memory."""

    content: str
    memory_type: MemoryType = "fact"
    evidence: str | None = None
    workspace: str | None = None


class ReplaceMemoryRequest(BaseModel):
    """Input for replacement of exactly one active memory."""

    memory_id: UUID
    content: str
    memory_type: MemoryType = "fact"
    evidence: str | None = None


class ForgetMemoryRequest(BaseModel):
    """Input for forgetting exactly one active memory."""

    memory_id: UUID
    reason: str | None = None


class MemorySubmissionReport(BaseModel):
    """Result of one explicit durable-memory operation."""

    outcome: Literal[
        "created", "reinforced", "replaced", "forgotten", "not_found", "inactive", "rejected"
    ]
    memory_id: UUID | None = None
    message: str
