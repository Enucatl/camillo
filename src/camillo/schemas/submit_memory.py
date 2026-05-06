from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

MemoryIntent = Literal["auto", "remember", "correct", "forget"]
DurableMemoryType = Literal[
    "semantic",
    "preference",
    "procedural",
    "relationship",
    "profile",
    "core",
]
MemoryScope = Literal["local", "shared", "global"]


class MemoryRelationshipClassification(BaseModel):
    """Validated LLM relationship judgment used by reconciliation policy.

    The schema keeps contradiction diagnosis explicit so a surface conflict does
    not automatically delete or supersede older context.
    """

    index: int
    relation: Literal[
        "confirms",
        "extends",
        "contradicts",
        "supersedes",
        "forgets",
        "unrelated",
        "duplicate",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    contradiction_type: Literal[
        "none",
        "direct_conflict",
        "temporal_shift",
        "context_shift",
        "scope_mismatch",
        "environment_difference",
        "preference_change",
        "implementation_change",
        "ambiguous",
    ] = "none"
    resolution: Literal[
        "keep_both",
        "supersede_old",
        "deprecate_old",
        "refine_old",
        "create_exception",
        "needs_review",
    ] = "keep_both"
    rationale: str | None = None
    old_memory_refinement: str | None = None
    new_memory_refinement: str | None = None


class SubmitMemoryRequest(BaseModel):
    """Request for durable memory reconciliation.

    The caller states intent, but the backend owns duplicate detection,
    contradiction handling, and lifecycle decisions.
    """

    namespace: str
    content: str
    intent: MemoryIntent = "auto"
    memory_type: DurableMemoryType | None = None
    scope: MemoryScope | None = None
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MemoryRelationReport(BaseModel):
    """Serializable relation created or considered during submission."""

    source_id: UUID | None = None
    target_id: UUID
    relation_type: str
    confidence: float
    rationale: str | None = None


class MemorySubmissionReport(BaseModel):
    """Transparent result of memory reconciliation."""

    outcome: Literal[
        "created",
        "reinforced",
        "superseded_old_memory",
        "deprecated_old_memory",
        "ignored_duplicate",
        "ignored_low_confidence",
        "no_related_memory_found",
    ]
    created_memory_id: UUID | None = None
    affected_memory_ids: list[UUID] = Field(default_factory=list)
    relations: list[MemoryRelationReport] = Field(default_factory=list)
    message: str
