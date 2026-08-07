import re

from camillo.cognitive.redaction import redact_secrets
from camillo.db.models import Memory
from camillo.interfaces import EmbeddingProvider, MemoryStoreProtocol


class IngestionService:
    """Capture every conversation turn as a redacted episode."""

    def __init__(self, memory_store: MemoryStoreProtocol, llm_service: EmbeddingProvider):
        """Initialize ingestion with storage and embedding dependencies."""
        self.memory_store = memory_store
        self.llm_service = llm_service

    async def ingest_interaction(
        self, user_msg: str, ai_msg: str, session_id: str | None, workspace: str | None
    ) -> Memory:
        """Redact and persist one automatic episode without graph side effects."""
        raw_content = redact_secrets(f"User:\n{user_msg}\n\nAssistant:\n{ai_msg}")
        if not raw_content.strip() or raw_content.strip() in {
            "User:\n\nAssistant:",
            "[REDACTED:PRIVATE_KEY]",
        }:
            raise ValueError("Content is empty after redaction")
        memory = await self.memory_store.insert_memory(
            raw_content=raw_content,
            embedding=await self.llm_service.get_embedding(raw_content, domain="ingest_embedding"),
            memory_type="episode",
            base_importance=score_interaction_importance(raw_content, ""),
            workspace=workspace,
            session_id=session_id,
        )
        return memory


def score_interaction_importance(user_msg: str, ai_msg: str) -> float:
    """Assign deterministic importance so routine ingestion does not need an LLM."""
    text = f"{user_msg}\n{ai_msg}".casefold()
    score = 0.45
    for pattern, boost in (
        (r"\b(prefer|always|never|remember|decided|must|constraint)\b", 0.2),
        (r"\b(correct|forget|deprecated|procedure|configure|deploy)\b", 0.15),
        (r"\b(api|database|postgres|docker|pytest|migration|security|bug)\b", 0.1),
    ):
        if re.search(pattern, text):
            score += boost
    return min(score + (0.05 if len(text) > 500 else 0), 1.0)
