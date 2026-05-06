import re

from camillo.db.models import Memory
from camillo.interfaces import EmbeddingProvider, GraphStoreProtocol, MemoryStoreProtocol


class IngestionService:
    """Coordinates interaction ingestion into memory and graph storage."""

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        graph_store: GraphStoreProtocol,
        llm_service: EmbeddingProvider,
    ):
        """Initialize ingestion with storage and AI provider dependencies."""
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.llm_service = llm_service

    async def ingest_interaction(
        self,
        namespace: str,
        user_msg: str,
        ai_msg: str,
        session_id: str | None,
    ) -> Memory:
        """Store a user/assistant interaction as an episodic memory.

        Args:
            namespace: Logical memory partition.
            user_msg: User side of the interaction.
            ai_msg: Assistant side of the interaction.
            session_id: Optional conversation id used to link adjacent turns.

        Returns:
            The inserted memory model.
        """
        raw_content = f"User:\n{user_msg}\n\nAssistant:\n{ai_msg}"
        base_importance = score_interaction_importance(user_msg, ai_msg)
        embedding = await self.llm_service.get_embedding(raw_content)
        previous_memory = None
        if session_id is not None:
            previous_memory = await self.memory_store.get_previous_memory_in_session(
                namespace,
                session_id,
            )

        memory = await self.memory_store.insert_memory(
            namespace=namespace,
            raw_content=raw_content,
            embedding=embedding,
            memory_type="episodic",
            base_importance=base_importance,
            session_id=session_id,
        )

        if previous_memory is not None:
            await self.graph_store.create_or_increment_edge(previous_memory.id, memory.id)

        return memory


def score_interaction_importance(user_msg: str, ai_msg: str) -> float:
    """Score an interaction deterministically before storing it.

    Ingestion should be cheap and reliable: every conversation turn still gets
    embedded for retrieval, but retention importance is local policy rather than
    an LLM call. Explicit preferences, decisions, corrections, procedures, and
    technical/project context get modest boosts over routine chat.

    Args:
        user_msg: User-side turn content.
        ai_msg: Assistant-side turn content.

    Returns:
        A clamped importance score from 0.0 to 1.0.
    """
    text = f"{user_msg}\n{ai_msg}".casefold()
    score = 0.45

    patterns: tuple[tuple[str, float], ...] = (
        (r"\b(prefer|prefers|preference|like|dislike|always|never)\b", 0.20),
        (r"\b(remember|decided|decision|must|require|required|constraint)\b", 0.20),
        (r"\b(correct|correction|instead|supersede|forget|deprecated)\b", 0.15),
        (r"\b(step|procedure|how to|run|install|configure|deploy)\b", 0.10),
        (r"\b(api|database|postgres|pgvector|docker|pytest|migration|service|repo)\b", 0.10),
        (r"\b(error|failed|bug|fix|security|urgent|blocker)\b", 0.10),
    )
    for pattern, boost in patterns:
        if re.search(pattern, text):
            score += boost

    if len(text) > 500:
        score += 0.05

    return max(0.0, min(score, 1.0))
