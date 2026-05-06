import os
import sys
from collections.abc import Callable
from typing import Annotated, Any

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.server import TransportSecuritySettings
    from mcp.types import ToolAnnotations
except ImportError:  # pragma: no cover - exercised only before dependency install

    class FastMCP:  # type: ignore[no-redef]
        """Minimal import-time fallback when the optional MCP package is absent."""

        def __init__(self, _name: str, **_kwargs: Any):
            """Accept the same constructor shape as FastMCP."""

        def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            """Return a decorator preserving registered functions."""
            return lambda function: function

        def run(self) -> None:
            """Fail at runtime with a dependency-focused error."""
            raise RuntimeError("Install the 'mcp' package to run the MCP server.")

    class TransportSecuritySettings:  # type: ignore[no-redef]
        """Minimal fallback preserving import-time configuration shape."""

        def __init__(self, **_kwargs: Any):
            """Accept the FastMCP transport security settings shape."""

    class ToolAnnotations:  # type: ignore[no-redef]
        """Minimal fallback preserving import-time tool annotation shape."""

        def __init__(self, **_kwargs: Any):
            """Accept the MCP tool annotation settings shape."""


from pydantic import Field

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.ingestion_service import IngestionService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import AsyncSessionLocal
from camillo.schemas.ingest import IngestResponse
from camillo.schemas.memory import McpRecalledMemory, McpRecallResponse, MemoryStatsResponse
from camillo.schemas.recall import ScoreBreakdown
from camillo.schemas.submit_memory import (
    DurableMemoryType,
    MemoryIntent,
    MemoryScope,
    MemorySubmissionReport,
)
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore

NamespaceArg = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Logical memory partition to read or write. Use a stable project, workspace, "
            "tenant, or user namespace to prevent unrelated memories from mixing."
        ),
    ),
]

RecallQueryArg = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Natural-language retrieval query. Include the concrete topic, project, "
            "constraint, or preference you want Camillo to recover."
        ),
    ),
]

TopKArg = Annotated[
    int | None,
    Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of primary direct matches to return. Omit to use the server default."
        ),
    ),
]

IncludeHebbianArg = Annotated[
    bool,
    Field(
        default=True,
        description=(
            "Whether to append graph-associated memories after the primary matches. "
            "Disable for strict direct retrieval."
        ),
    ),
]

IncludeSharedArg = Annotated[
    bool,
    Field(
        default=True,
        description=(
            "Whether recall may include shared/global memories from other namespaces. "
            "Disable for strict namespace-local recall."
        ),
    ),
]

UserMessageArg = Annotated[
    str,
    Field(
        min_length=1,
        description="User-side content from the conversation turn to store as episodic memory.",
    ),
]

AssistantMessageArg = Annotated[
    str,
    Field(
        min_length=1,
        description="Assistant-side content from the same conversation turn.",
    ),
]

SessionIdArg = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Optional stable conversation or task identifier. Turns with the same session "
            "are linked in the memory graph."
        ),
    ),
]

MemoryContentArg = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Durable memory candidate to reconcile, such as a user preference, project "
            "constraint, correction, procedure, or profile fact."
        ),
    ),
]

EvidenceArg = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Optional source text or rationale supporting the memory. Stored as metadata "
            "for auditability."
        ),
    ),
]

ConfidenceArg = Annotated[
    float | None,
    Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional caller confidence. Omit to let Camillo use its default.",
    ),
]

MemoryScopeArg = Annotated[
    MemoryScope | None,
    Field(
        default=None,
        description=(
            "Optional reuse scope for durable memories: local, shared, or global. "
            "Omit to derive scope from memory_type."
        ),
    ),
]

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

RECONCILE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)


def _mcp_allowed_hosts() -> list[str]:
    """Build the FastMCP host allowlist without disabling rebinding protection.

    FastMCP protects localhost-bound servers by validating Host headers. Camillo
    is mounted behind Traefik, so deployments need to add the public reverse
    proxy hostname while preserving the localhost defaults used for direct runs.

    Returns:
        Host header patterns accepted by FastMCP transport security.
    """
    app_host = settings.app_name.lower()
    defaults = [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
        app_host,
        f"{app_host}:*",
    ]
    configured = [
        host.strip() for host in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    configured_with_ports: list[str] = []
    for host in configured:
        configured_with_ports.append(host)
        if not host.startswith("[") and ":" not in host:
            configured_with_ports.append(f"{host}:*")
    return list(dict.fromkeys(defaults + configured_with_ports))


mcp = FastMCP(
    "cognitive-memory-stack",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_mcp_allowed_hosts(),
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ],
    ),
)


@mcp.tool(
    title="Recall Active Memories",
    description=(
        "Retrieve relevant active memories from one namespace for a concrete query. "
        "Use this before answering when prior project context, user preferences, or "
        "operational constraints may affect the response."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    meta={
        "when_to_use": [
            "Before answering a question that may depend on prior user or project context.",
            "When a task references a project, preference, constraint, decision, "
            "or past instruction.",
            "For project memory, use a repo-scoped namespace like repo:<repo_name> "
            "instead of the service name camillo.",
        ],
        "when_not_to_use": [
            "For storing new information; use record_interaction or submit_memory instead.",
            "For global search across unrelated namespaces.",
        ],
        "side_effects": [],
        "returns": "query, namespace, and ranked memories with score provenance.",
    },
)
async def recall_memory(
    query: RecallQueryArg,
    namespace: NamespaceArg,
    top_k: TopKArg = None,
    include_hebbian: IncludeHebbianArg = True,
    include_shared: IncludeSharedArg = True,
) -> McpRecallResponse:
    """Recall relevant active memories through the Phase 2 pipeline."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            graph_store = GraphStore(db)
            llm_service = LiteLLMService()
            service = RecallService(memory_store, graph_store, llm_service)
            candidates = await service.recall(
                namespace=namespace,
                query=query,
                top_k=top_k or settings.recall_top_k,
                include_hebbian=include_hebbian,
                include_shared=include_shared,
            )
            await db.commit()
            return McpRecallResponse(
                query=query,
                namespace=namespace,
                memories=[
                    McpRecalledMemory(
                        id=candidate.memory.id,
                        namespace=candidate.memory.namespace,
                        scope=candidate.memory.scope,
                        raw_content=candidate.memory.raw_content,
                        type=candidate.memory.type,
                        base_importance=candidate.memory.base_importance,
                        score=candidate.final_score or 0.0,
                        source=candidate.source,
                        linked_from=candidate.linked_from,
                        edge_weight=candidate.edge_weight,
                        score_breakdown=ScoreBreakdown(
                            retrieval_score=candidate.retrieval_score,
                            rerank_score=candidate.rerank_score,
                            activation_score=candidate.activation_score or 0.0,
                            scope_affinity_score=candidate.scope_affinity_score or 0.0,
                            final_score=candidate.final_score or 0.0,
                            vector_score=candidate.vector_score,
                            text_score=candidate.text_score,
                            rrf_score=candidate.rrf_score,
                        ),
                    )
                    for candidate in candidates
                ],
            )
        except Exception:
            await db.rollback()
            raise


@mcp.tool(
    title="Record Interaction",
    description=(
        "Store one user/assistant conversation turn as episodic memory. Use this for "
        "raw interaction history that may become useful later; Camillo scores the turn, "
        "embeds it, and links adjacent turns that share a session_id."
    ),
    annotations=WRITE_ANNOTATIONS,
    meta={
        "when_to_use": [
            "After a meaningful exchange that should be available for future recall.",
            "When preserving both user wording and assistant response matters.",
        ],
        "when_not_to_use": [
            "For concise durable facts or corrections; use submit_memory instead.",
            "For read-only lookup; use recall_memory or memory_stats.",
        ],
        "side_effects": [
            "Creates an episodic memory row.",
            "May create or strengthen a session adjacency edge.",
        ],
        "returns": "created memory id, namespace, memory type, and computed importance score.",
    },
)
async def record_interaction(
    namespace: NamespaceArg,
    user_msg: UserMessageArg,
    ai_msg: AssistantMessageArg,
    session_id: SessionIdArg = None,
) -> IngestResponse:
    """Record a raw user/assistant exchange as episodic memory."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            graph_store = GraphStore(db)
            llm_service = LiteLLMService()
            service = IngestionService(memory_store, graph_store, llm_service)
            memory = await service.ingest_interaction(namespace, user_msg, ai_msg, session_id)
            await db.commit()
            return IngestResponse(
                memory_id=memory.id,
                namespace=memory.namespace,
                type=memory.type,
                base_importance=memory.base_importance,
            )
        except Exception:
            await db.rollback()
            raise


@mcp.tool(
    title="Submit Durable Memory",
    description=(
        "Reconcile a durable memory candidate against related active memories. Use this "
        "for explicit preferences, corrections, procedures, project constraints, profile "
        "facts, or forget requests. Camillo detects duplicates and contextual conflicts "
        "before creating, reinforcing, superseding, or deprecating memories."
    ),
    annotations=RECONCILE_ANNOTATIONS,
    meta={
        "when_to_use": [
            "When the user asks to remember, correct, forget, or preserve a durable fact.",
            "When a compact semantic memory is better than storing the whole conversation turn.",
        ],
        "when_not_to_use": [
            "For raw conversation logging; use record_interaction.",
            "For retrieval without mutation; use recall_memory or memory_stats.",
        ],
        "side_effects": [
            "May create a durable memory.",
            "May reinforce, supersede, or deprecate related active memories.",
            "May create semantic relation rows between memories.",
        ],
        "valid_intents": ["auto", "remember", "correct", "forget"],
        "valid_memory_types": [
            "semantic",
            "preference",
            "procedural",
            "relationship",
            "profile",
            "core",
        ],
        "returns": "transparent reconciliation report with outcome, affected ids, and relations.",
    },
)
async def submit_memory(
    namespace: NamespaceArg,
    content: MemoryContentArg,
    intent: MemoryIntent = "auto",
    memory_type: DurableMemoryType | None = None,
    scope: MemoryScopeArg = None,
    evidence: EvidenceArg = None,
    confidence: ConfidenceArg = None,
) -> MemorySubmissionReport:
    """Submit durable memory through reconciliation policy."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            graph_store = GraphStore(db)
            relation_store = RelationStore(db)
            llm_service = LiteLLMService()
            recall_service = RecallService(memory_store, graph_store, llm_service)
            service = MemoryReconciliationService(
                memory_store,
                relation_store,
                recall_service,
                llm_service,
            )
            report = await service.submit_memory(
                namespace=namespace,
                content=content,
                intent=intent,
                memory_type=memory_type,
                scope=scope,
                evidence=evidence,
                confidence=confidence,
            )
            await db.commit()
            return report
        except Exception:
            await db.rollback()
            raise


@mcp.tool(
    title="Memory Stats",
    description=(
        "Return operational memory counts for one namespace. Use this to inspect whether "
        "a namespace has active, deprecated, superseded, episodic, or durable memories "
        "before deciding whether to recall or write memory."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    meta={
        "when_to_use": [
            "Before diagnosing whether a namespace has stored memory.",
            "When checking counts by memory type or lifecycle status.",
            "For project memory, use a repo-scoped namespace like repo:<repo_name> "
            "instead of the service name camillo.",
        ],
        "when_not_to_use": [
            "For retrieving memory content; use recall_memory.",
            "For storing or reconciling memory; use record_interaction or submit_memory.",
        ],
        "side_effects": [],
        "returns": "namespace, total count, counts by memory type, and counts by status.",
    },
)
async def memory_stats(namespace: NamespaceArg) -> MemoryStatsResponse:
    """Return operational memory counts for a namespace."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            stats = await memory_store.memory_stats(namespace)
            await db.commit()
            return MemoryStatsResponse.model_validate(stats)
        except Exception:
            await db.rollback()
            raise


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
