from uuid import UUID

from pydantic import BaseModel, Field


class RecallRequest(BaseModel):
    """Keep recall inputs explicit as graph expansion becomes optional.

    Clients can preserve Phase 1 behavior by omitting the new fields while newer
    callers can opt out of associative context per request.
    """

    namespace: str
    query: str
    top_k: int | None = None
    include_hebbian: bool = True
    include_shared: bool = True


class ScoreBreakdown(BaseModel):
    """Make ranking decisions inspectable without changing the top-level score.

    The API needs enough provenance to debug relevance, activation, and graph
    effects while keeping backward compatibility for existing score consumers.
    """

    retrieval_score: float = Field(ge=0.0)
    rerank_score: float | None = None
    activation_score: float = Field(ge=0.0)
    scope_affinity_score: float = Field(ge=0.0)
    final_score: float = Field(ge=0.0)
    vector_score: float | None = None
    text_score: float | None = None
    rrf_score: float | None = None


class RecalledMemory(BaseModel):
    """Return memory content with the context needed to interpret its source.

    Primary and Hebbian memories share one response shape so clients can render a
    single list while still distinguishing direct hits from graph additions.
    """

    id: UUID
    namespace: str
    scope: str
    raw_content: str
    type: str
    base_importance: float
    access_count: int
    score: float
    score_breakdown: ScoreBreakdown
    source: str = "primary"
    linked_from: UUID | None = None
    edge_weight: float | None = None


class RecallResponse(BaseModel):
    """Group recalled memories with the query metadata used to produce them.

    Echoing namespace and query gives API callers a stable envelope for tracing
    and caching recall responses.
    """

    query: str
    namespace: str
    memories: list[RecalledMemory]
