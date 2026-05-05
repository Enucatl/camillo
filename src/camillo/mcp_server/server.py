import os
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.server import TransportSecuritySettings
except ImportError:  # pragma: no cover - exercised only before dependency install

    class FastMCP:  # type: ignore[no-redef]
        """Minimal import-time fallback when the optional MCP package is absent."""

        def __init__(self, _name: str, **_kwargs: Any):
            """Accept the same constructor shape as FastMCP."""

        def tool(self):
            """Return a decorator preserving registered functions."""
            return lambda function: function

        def run(self) -> None:
            """Fail at runtime with a dependency-focused error."""
            raise RuntimeError("Install the 'mcp' package to run the MCP server.")

    class TransportSecuritySettings:  # type: ignore[no-redef]
        """Minimal fallback preserving import-time configuration shape."""

        def __init__(self, **_kwargs: Any):
            """Accept the FastMCP transport security settings shape."""


from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.ingestion_service import IngestionService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import AsyncSessionLocal
from camillo.schemas.recall import RecalledMemory, ScoreBreakdown
from camillo.settings import settings
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore


def _mcp_allowed_hosts() -> list[str]:
    """Build the FastMCP host allowlist without disabling rebinding protection.

    FastMCP protects localhost-bound servers by validating Host headers. Camillo
    is mounted behind Traefik, so deployments need to add the public reverse
    proxy hostname while preserving the localhost defaults used for direct runs.

    Returns:
        Host header patterns accepted by FastMCP transport security.
    """
    defaults = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    configured = [
        host.strip() for host in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    return defaults + configured


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


@mcp.tool()
async def recall_memory(
    query: str,
    namespace: str,
    top_k: int | None = None,
    include_hebbian: bool = True,
) -> dict[str, Any]:
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
            )
            await db.commit()
            return {
                "query": query,
                "namespace": namespace,
                "memories": [
                    RecalledMemory(
                        id=candidate.memory.id,
                        namespace=candidate.memory.namespace,
                        raw_content=candidate.memory.raw_content,
                        type=candidate.memory.type,
                        base_importance=candidate.memory.base_importance,
                        access_count=candidate.memory.access_count,
                        score=candidate.final_score or 0.0,
                        source=candidate.source,
                        linked_from=candidate.linked_from,
                        edge_weight=candidate.edge_weight,
                        score_breakdown=ScoreBreakdown(
                            retrieval_score=candidate.retrieval_score,
                            rerank_score=candidate.rerank_score,
                            activation_score=candidate.activation_score or 0.0,
                            final_score=candidate.final_score or 0.0,
                            vector_score=candidate.vector_score,
                            text_score=candidate.text_score,
                            rrf_score=candidate.rrf_score,
                        ),
                    ).model_dump(mode="json")
                    for candidate in candidates
                ],
            }
        except Exception:
            await db.rollback()
            raise


@mcp.tool()
async def record_interaction(
    namespace: str,
    user_msg: str,
    ai_msg: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Record a raw user/assistant exchange as episodic memory."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            graph_store = GraphStore(db)
            llm_service = LiteLLMService()
            service = IngestionService(memory_store, graph_store, llm_service)
            memory = await service.ingest_interaction(namespace, user_msg, ai_msg, session_id)
            await db.commit()
            return {
                "memory_id": str(memory.id),
                "namespace": memory.namespace,
                "type": memory.type,
                "base_importance": memory.base_importance,
            }
        except Exception:
            await db.rollback()
            raise


@mcp.tool()
async def submit_memory(
    namespace: str,
    content: str,
    intent: str = "auto",
    memory_type: str | None = None,
    evidence: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
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
                evidence=evidence,
                confidence=confidence,
            )
            await db.commit()
            return report.model_dump(mode="json")
        except Exception:
            await db.rollback()
            raise


@mcp.tool()
async def memory_stats(namespace: str) -> dict[str, Any]:
    """Return operational memory counts for a namespace."""
    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            stats = await memory_store.memory_stats(namespace)
            await db.commit()
            return stats
        except Exception:
            await db.rollback()
            raise


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
