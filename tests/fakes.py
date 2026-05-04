import hashlib
import random
import re
from datetime import UTC, datetime
from itertools import combinations
from typing import Any
from uuid import UUID, uuid4

from camillo.db.models import Memory


def tokenize(text: str) -> set[str]:
    """Keep fake lexical scoring aligned with punctuation-heavy memory text.

    Args:
        text: Query or memory content.

    Returns:
        Lower-cased alphanumeric terms used by deterministic test scoring.
    """
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def synthetic_embedding(text: str, dim: int = 32) -> list[float]:
    """Generate repeatable embeddings so tests exercise ranking deterministically.

    Args:
        text: Content to map into a pseudo-vector.
        dim: Embedding dimension to produce.

    Returns:
        A stable pseudo-random vector for the given text.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Mirror production vector scoring without requiring a database.

    Args:
        left: First embedding vector.
        right: Second embedding vector.

    Returns:
        Cosine similarity, or `0.0` for zero vectors.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def make_memory(
    raw_content: str,
    *,
    namespace: str = "test",
    session_id: str | None = None,
    embedding: list[float] | None = None,
    base_importance: float = 0.5,
    memory_type: str = "episodic",
) -> Memory:
    """Create model-shaped memories so tests cover real service contracts.

    Args:
        raw_content: Memory text.
        namespace: Logical partition assigned to the memory.
        session_id: Optional conversation/session grouping.
        embedding: Optional explicit vector for ranking tests.
        base_importance: ACT-R base importance value.
        memory_type: Memory category to assign.

    Returns:
        An unsaved `Memory` instance with active status.
    """
    now = datetime.now(UTC)
    return Memory(
        id=uuid4(),
        namespace=namespace,
        session_id=session_id,
        raw_content=raw_content,
        embedding=embedding or synthetic_embedding(raw_content),
        type=memory_type,
        status="active",
        confidence=0.8,
        source=None,
        superseded_by=None,
        deprecated_at=None,
        status_reason=None,
        base_importance=base_importance,
        access_count=0,
        created_at=now,
        last_accessed_at=now,
        metadata_json={},
    )


class FakeLLMService:
    """Provide deterministic AI behavior so recall tests isolate pipeline logic."""

    def __init__(self, dim: int = 32, valence: float = 0.7):
        """Capture calls while keeping outputs stable.

        Args:
            dim: Embedding size for fake vectors.
            valence: Importance score returned by ingestion tests.
        """
        self.dim = dim
        self.valence = valence
        self.scored: list[str] = []
        self.embedded: list[str] = []
        self.reranked: list[tuple[str, list[str]]] = []
        self.classifications: list[Any] | None = None

    async def score_valence(self, raw_content: str) -> float:
        """Avoid provider calls while still verifying ingestion wiring.

        Args:
            raw_content: Memory text that would be scored by an LLM.

        Returns:
            The configured deterministic valence score.
        """
        self.scored.append(raw_content)
        return self.valence

    async def get_embedding(self, text: str) -> list[float]:
        """Record embedding requests so tests can assert query flow.

        Args:
            text: Text to embed.

        Returns:
            A deterministic pseudo-embedding.
        """
        self.embedded.append(text)
        return synthetic_embedding(text, self.dim)

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Make reranking semantic enough for tests without provider variance.

        Args:
            query: Recall query.
            documents: Candidate texts.

        Returns:
            Query-term overlap scores in document order.
        """
        self.reranked.append((query, documents))
        terms = tokenize(query)
        return [len(terms & tokenize(document)) / max(len(terms), 1) for document in documents]

    async def classify_memory_relationships(
        self,
        intent: str,
        new_content: str,
        existing_memories: list[Memory],
    ) -> list[Any]:
        """Return caller-configured relationship judgments for reconciliation tests.

        Args:
            intent: Submission intent.
            new_content: Candidate durable memory.
            existing_memories: Related active memories.

        Returns:
            Preconfigured classifications or unrelated fallbacks.
        """
        if self.classifications is not None:
            return self.classifications
        from camillo.schemas.submit_memory import MemoryRelationshipClassification

        return [
            MemoryRelationshipClassification(
                index=index,
                relation="unrelated",
                confidence=0.0,
                rationale="classification unavailable",
            )
            for index, _memory in enumerate(existing_memories)
        ]


class FakeMemoryStore:
    """Keep store tests in memory while preserving production method semantics."""

    def __init__(self, memories: list[Memory] | None = None):
        """Start tests from a caller-controlled memory set.

        Args:
            memories: Optional initial memory rows.
        """
        self.memories = memories or []
        self.marked_accessed: list[UUID] = []

    async def insert_memory(
        self,
        namespace: str,
        raw_content: str,
        embedding: list[float],
        memory_type: str,
        base_importance: float,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        confidence: float | None = None,
        source: str | None = None,
        status: str = "active",
    ) -> Memory:
        """Append memories so ingestion tests can observe persistence effects.

        Args:
            namespace: Logical memory partition.
            raw_content: Memory text.
            embedding: Precomputed embedding.
            memory_type: Memory category.
            base_importance: Long-term importance score.
            session_id: Optional session grouping.
            metadata: Optional structured metadata.

        Returns:
            The inserted fake memory.
        """
        memory = make_memory(
            raw_content,
            namespace=namespace,
            session_id=session_id,
            embedding=embedding,
            base_importance=base_importance,
            memory_type=memory_type,
        )
        memory.status = status
        memory.confidence = confidence if confidence is not None else 0.8
        memory.source = source
        memory.metadata_json = metadata or {}
        self.memories.append(memory)
        return memory

    async def get_previous_memory_in_session(
        self, namespace: str, session_id: str
    ) -> Memory | None:
        """Support ingestion adjacency without needing database ordering.

        Args:
            namespace: Partition to search.
            session_id: Session whose newest memory is needed.

        Returns:
            Last active memory inserted for that namespace/session, if any.
        """
        matches = [
            memory
            for memory in self.memories
            if memory.namespace == namespace
            and memory.session_id == session_id
            and memory.status == "active"
        ]
        return matches[-1] if matches else None

    async def vector_candidates(
        self,
        namespace: str,
        embedding: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Let recall tests exercise semantic retrieval ordering.

        Args:
            namespace: Partition to search.
            embedding: Query embedding.
            limit: Maximum candidates to return.

        Returns:
            Active same-namespace memories sorted by cosine similarity.
        """
        matches = [
            (memory, cosine_similarity(memory.embedding, embedding))
            for memory in self.memories
            if memory.namespace == namespace and memory.status == "active"
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    async def full_text_search_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Let recall tests exercise lexical retrieval and RRF behavior.

        Args:
            namespace: Partition to search.
            query: Raw recall query.
            limit: Maximum candidates to return.

        Returns:
            Active same-namespace memories with positive token overlap.
        """
        terms = tokenize(query)
        matches = []
        for memory in self.memories:
            if memory.namespace != namespace or memory.status != "active":
                continue
            words = tokenize(memory.raw_content)
            score = len(terms & words) / max(len(terms), 1)
            if score > 0.0:
                matches.append((memory, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    async def get_memories_by_ids(
        self,
        memory_ids: list[UUID],
        *,
        active_only: bool = True,
    ) -> list[Memory]:
        """Hydrate graph neighbors while matching production active filtering.

        Args:
            memory_ids: IDs to fetch.
            active_only: Whether inactive memories should be excluded.

        Returns:
            Matching fake memories.
        """
        memory_id_set = set(memory_ids)
        return [
            memory
            for memory in self.memories
            if memory.id in memory_id_set and (not active_only or memory.status == "active")
        ]

    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        """Mutate fake memories so reinforcement assertions see side effects.

        Args:
            memory_ids: Memories surfaced by recall.
        """
        self.marked_accessed.extend(memory_ids)
        for memory in self.memories:
            if memory.id in memory_ids:
                memory.access_count += 1
                memory.last_accessed_at = datetime.now(UTC)

    async def update_memory_status(
        self,
        memory_id: UUID,
        status: str,
        *,
        reason: str | None = None,
        superseded_by: UUID | None = None,
    ) -> Memory | None:
        """Apply fake lifecycle updates for reconciliation tests.

        Args:
            memory_id: Memory to update.
            status: New lifecycle status.
            reason: Optional status reason.
            superseded_by: Replacement memory id when applicable.

        Returns:
            Updated fake memory or `None`.
        """
        for memory in self.memories:
            if memory.id != memory_id:
                continue
            memory.status = status
            memory.status_reason = reason
            if status in {"deprecated", "superseded"}:
                memory.deprecated_at = datetime.now(UTC)
                memory.base_importance = min(memory.base_importance, 0.2)
            if superseded_by is not None:
                memory.superseded_by = superseded_by
            return memory
        return None

    async def reinforce_memory(
        self,
        memory_id: UUID,
        *,
        increment_access: bool = True,
        importance_boost: float = 0.05,
    ) -> Memory | None:
        """Strengthen fake memories when reconciliation avoids duplicates.

        Args:
            memory_id: Memory to reinforce.
            increment_access: Whether access count should increase.
            importance_boost: Importance increment.

        Returns:
            Updated fake memory or `None`.
        """
        for memory in self.memories:
            if memory.id != memory_id:
                continue
            if increment_access:
                memory.access_count += 1
            memory.base_importance = min(memory.base_importance + importance_boost, 1.0)
            memory.last_accessed_at = datetime.now(UTC)
            return memory
        return None

    async def memory_stats(self, namespace: str) -> dict[str, Any]:
        """Count fake memories using production response shape.

        Args:
            namespace: Partition to count.

        Returns:
            Namespace stats.
        """
        matches = [memory for memory in self.memories if memory.namespace == namespace]
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for memory in matches:
            by_type[memory.type] = by_type.get(memory.type, 0) + 1
            by_status[memory.status] = by_status.get(memory.status, 0) + 1
        return {
            "namespace": namespace,
            "total": len(matches),
            "by_type": by_type,
            "by_status": by_status,
        }


class FakeGraphStore:
    """Model Hebbian behavior in memory for recall and ingestion tests."""

    def __init__(self):
        """Create an empty canonical undirected edge map."""
        self.edges: dict[tuple[UUID, UUID], float] = {}

    async def create_or_increment_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        increment: float = 1.0,
    ) -> None:
        """Mirror production canonical edge updates without SQLAlchemy.

        Args:
            source_id: One memory in the association.
            target_id: The other memory in the association.
            increment: Weight to add to the pair.
        """
        if source_id == target_id:
            return
        edge = tuple(sorted((source_id, target_id), key=str))
        self.edges[edge] = self.edges.get(edge, 0.0) + increment

    async def reinforce_clique(self, memory_ids: list[UUID], increment: float = 1.0) -> None:
        """Let tests verify co-recall reinforcement without a database.

        Args:
            memory_ids: Memories returned together.
            increment: Weight to add to each unique pair.
        """
        for source_id, target_id in combinations(dict.fromkeys(memory_ids), 2):
            await self.create_or_increment_edge(source_id, target_id, increment)

    async def get_strong_neighbors(
        self,
        memory_ids: list[UUID],
        *,
        min_weight: float,
        limit_per_source: int,
    ) -> list[tuple[UUID, UUID, float]]:
        """Expose only strong fake edges so Hebbian tests match production intent.

        Args:
            memory_ids: Primary memories anchoring graph expansion.
            min_weight: Minimum edge weight to include.
            limit_per_source: Maximum neighbors per primary memory.

        Returns:
            Source, neighbor, and weight tuples ordered by strength.
        """
        source_ids = list(dict.fromkeys(memory_ids))
        source_set = set(source_ids)
        best_by_neighbor: dict[UUID, tuple[UUID, UUID, float]] = {}

        for source_id in source_ids:
            links: list[tuple[UUID, UUID, float]] = []
            for (left_id, right_id), weight in self.edges.items():
                if weight < min_weight:
                    continue
                if left_id == source_id and right_id not in source_set:
                    links.append((source_id, right_id, weight))
                elif right_id == source_id and left_id not in source_set:
                    links.append((source_id, left_id, weight))

            links.sort(key=lambda item: item[2], reverse=True)
            for link in links[:limit_per_source]:
                current = best_by_neighbor.get(link[1])
                if current is None or link[2] > current[2]:
                    best_by_neighbor[link[1]] = link

        links = list(best_by_neighbor.values())
        links.sort(key=lambda item: item[2], reverse=True)
        return links
