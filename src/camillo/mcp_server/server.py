import os
import sys
from collections.abc import Callable
from typing import Annotated, Any

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.server import TransportSecuritySettings
    from mcp.types import ToolAnnotations
except ImportError:  # pragma: no cover

    class FastMCP:  # type: ignore[no-redef]
        """Import fallback for environments without MCP installed."""

        def __init__(self, _name: str, **_kwargs: Any):
            pass

        def tool(self, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return lambda f: f

        def run(self, **_kwargs: Any) -> None:
            raise RuntimeError("Install the mcp package.")

    class TransportSecuritySettings:  # type: ignore[no-redef]
        """Fallback transport settings."""

        def __init__(self, **_kwargs: Any):
            pass

    class ToolAnnotations:  # type: ignore[no-redef]
        """Fallback tool annotations."""

        def __init__(self, **_kwargs: Any):
            pass


from pydantic import Field

from camillo.ai.llm_service import get_inference_service
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import AsyncSessionLocal
from camillo.schemas.submit_memory import MemorySubmissionReport
from camillo.settings import settings
from camillo.stores.memory_store import MemoryStore

Query = Annotated[str, Field(min_length=1, description="Natural-language memory query.")]
Workspace = Annotated[
    str | None, Field(default=None, description="Optional workspace ranking hint.")
]
Content = Annotated[str, Field(min_length=1, description="Memory content.")]
MemoryId = Annotated[str, Field(min_length=1, description="One explicit memory UUID.")]


def _mcp_allowed_hosts() -> list[str]:
    """Allow local development and the configured reverse-proxy hostname."""
    defaults = [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
    ]
    configured = [
        host.strip() for host in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    expanded = []
    for host in configured:
        expanded.append(host)
        if not host.startswith("[") and ":" not in host:
            expanded.append(f"{host}:*")
    return list(dict.fromkeys(defaults + expanded))


mcp = FastMCP(
    "camillo",
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
READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)


def _service(db: Any) -> MemoryReconciliationService:
    """Build the durable-memory boundary for one MCP transaction."""
    store = MemoryStore(db)
    provider = get_inference_service()
    return MemoryReconciliationService(store, RecallService(store, provider), provider)


@mcp.tool(title="Recall Active Memories", annotations=WRITE)
async def recall_memory(
    query: Query, top_k: int | None = None, workspace: Workspace = None
) -> dict[str, Any]:
    """Recall active memories and reinforce their access metadata."""
    async with AsyncSessionLocal() as db:
        service = RecallService(MemoryStore(db), get_inference_service())
        candidates = await service.recall(query, top_k or settings.recall_top_k, workspace)
        await db.commit()
        return {
            "query": query,
            "workspace": workspace,
            "memories": [_candidate_dict(c) for c in candidates],
        }


@mcp.tool(title="Remember Memory", annotations=WRITE)
async def remember_memory(
    content: Content,
    memory_type: str = "fact",
    evidence: str | None = None,
    workspace: Workspace = None,
) -> MemorySubmissionReport:
    """Create or reinforce one durable memory."""
    async with AsyncSessionLocal() as db:
        report = await _service(db).remember_memory(content, memory_type, evidence, workspace)
        await db.commit()
        return report


@mcp.tool(title="Replace Memory", annotations=DESTRUCTIVE)
async def replace_memory(
    memory_id: MemoryId, content: Content, memory_type: str = "fact", evidence: str | None = None
) -> MemorySubmissionReport:
    """Replace one explicit active memory."""
    from uuid import UUID

    async with AsyncSessionLocal() as db:
        report = await _service(db).replace_memory(UUID(memory_id), content, memory_type, evidence)
        await db.commit()
        return report


@mcp.tool(title="Forget Memory", annotations=DESTRUCTIVE)
async def forget_memory(memory_id: MemoryId, reason: str | None = None) -> MemorySubmissionReport:
    """Forget one explicit active memory."""
    from uuid import UUID

    async with AsyncSessionLocal() as db:
        report = await _service(db).forget_memory(UUID(memory_id), reason)
        await db.commit()
        return report


@mcp.tool(title="Memory Stats", annotations=READ_ONLY)
async def memory_stats(workspace: Workspace = None) -> dict[str, Any]:
    """Return corpus counts, optionally filtered by workspace."""
    async with AsyncSessionLocal() as db:
        stats = await MemoryStore(db).memory_stats(workspace)
        await db.commit()
        return {"workspace": workspace, **stats}


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    """Serialize internal candidate provenance for MCP clients."""
    return {
        "id": str(candidate.memory.id),
        "workspace": candidate.memory.workspace,
        "raw_content": candidate.memory.raw_content,
        "type": candidate.memory.type,
        "score": candidate.final_score or 0.0,
        "score_breakdown": {
            "retrieval_score": candidate.retrieval_score,
            "rerank_score": candidate.rerank_score,
            "activation_score": candidate.activation_score or 0.0,
            "workspace_affinity_score": candidate.workspace_affinity_score or 0.0,
            "final_score": candidate.final_score or 0.0,
        },
    }


if __name__ == "__main__":
    mcp.run(transport=sys.argv[1] if len(sys.argv) > 1 else os.getenv("MCP_TRANSPORT", "stdio"))
