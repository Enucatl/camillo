from uuid import UUID

from pydantic import BaseModel, Field


class RecallRequest(BaseModel):
    """Input for corpus recall; workspace is only a ranking hint."""

    query: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    workspace: str | None = None


class ScoreBreakdown(BaseModel):
    """Expose the fixed relevance, activation, and workspace score inputs."""

    retrieval_score: float = Field(ge=0.0)
    rerank_score: float | None = None
    activation_score: float = Field(ge=0.0)
    workspace_affinity_score: float = Field(ge=0.0)
    final_score: float = Field(ge=0.0)
    vector_score: float | None = None
    text_score: float | None = None
    rrf_score: float | None = None


class RecalledMemory(BaseModel):
    """One ranked memory and score provenance."""

    id: UUID
    workspace: str | None
    raw_content: str
    type: str
    base_importance: float
    access_count: int
    score: float
    score_breakdown: ScoreBreakdown


class RecallResponse(BaseModel):
    """Stable recall response envelope."""

    query: str
    workspace: str | None
    memories: list[RecalledMemory]
