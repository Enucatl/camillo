import hashlib
import random
from datetime import UTC, datetime
from itertools import combinations
from typing import Any
from uuid import UUID, uuid4

from camillo.db.models import Memory


def synthetic_embedding(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
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
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=uuid4(),
        namespace=namespace,
        session_id=session_id,
        raw_content=raw_content,
        embedding=embedding or synthetic_embedding(raw_content),
        type="episodic",
        status="active",
        base_importance=base_importance,
        access_count=0,
        created_at=now,
        last_accessed_at=now,
        metadata_json={},
    )


class FakeLLMService:
    def __init__(self, dim: int = 32, valence: float = 0.7):
        self.dim = dim
        self.valence = valence
        self.scored: list[str] = []
        self.embedded: list[str] = []

    async def score_valence(self, raw_content: str) -> float:
        self.scored.append(raw_content)
        return self.valence

    async def get_embedding(self, text: str) -> list[float]:
        self.embedded.append(text)
        return synthetic_embedding(text, self.dim)

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 for _ in documents]


class FakeMemoryStore:
    def __init__(self, memories: list[Memory] | None = None):
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
    ) -> Memory:
        memory = make_memory(
            raw_content,
            namespace=namespace,
            session_id=session_id,
            embedding=embedding,
            base_importance=base_importance,
        )
        memory.type = memory_type
        memory.metadata_json = metadata or {}
        self.memories.append(memory)
        return memory

    async def get_previous_memory_in_session(
        self, namespace: str, session_id: str
    ) -> Memory | None:
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
        matches = [
            (memory, cosine_similarity(memory.embedding, embedding))
            for memory in self.memories
            if memory.namespace == namespace and memory.status == "active"
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    async def fts_candidates(
        self,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        terms = set(query.lower().split())
        matches = []
        for memory in self.memories:
            if memory.namespace != namespace or memory.status != "active":
                continue
            words = set(memory.raw_content.lower().split())
            score = len(terms & words) / max(len(terms), 1)
            matches.append((memory, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    async def mark_accessed(self, memory_ids: list[UUID]) -> None:
        self.marked_accessed.extend(memory_ids)
        for memory in self.memories:
            if memory.id in memory_ids:
                memory.access_count += 1
                memory.last_accessed_at = datetime.now(UTC)


class FakeGraphStore:
    def __init__(self):
        self.edges: dict[tuple[UUID, UUID], float] = {}

    async def create_or_increment_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        increment: float = 1.0,
    ) -> None:
        if source_id == target_id:
            return
        edge = tuple(sorted((source_id, target_id), key=str))
        self.edges[edge] = self.edges.get(edge, 0.0) + increment

    async def reinforce_clique(self, memory_ids: list[UUID], increment: float = 1.0) -> None:
        for source_id, target_id in combinations(dict.fromkeys(memory_ids), 2):
            await self.create_or_increment_edge(source_id, target_id, increment)
