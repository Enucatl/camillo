from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from camillo.cognitive.ingestion_service import score_interaction_importance
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.redaction import redact_secrets
from camillo.schemas.recall import RecallRequest
from camillo.schemas.submit_memory import RememberMemoryRequest


def test_redaction_removes_credentials_before_storage() -> None:
    content = "token=secret password=hunter2 https://user:pass@example.test"
    redacted = redact_secrets(content)
    assert "secret" not in redacted
    assert "hunter2" not in redacted
    assert "pass@example" not in redacted
    assert "[REDACTED:API_TOKEN]" in redacted
    assert "[REDACTED:PASSWORD]" in redacted


def test_new_contract_defaults_and_types() -> None:
    assert RememberMemoryRequest(content="use pytest").memory_type == "fact"
    assert RecallRequest(query="pytest").workspace is None


def test_importance_is_deterministic() -> None:
    assert score_interaction_importance(
        "remember this constraint", "use pytest"
    ) == score_interaction_importance("remember this constraint", "use pytest")


class Provider:
    async def get_embedding(self, _text: str) -> list[float]:
        return [1.0]

    async def rerank_results(self, _query: str, documents: list[str]) -> list[float]:
        return [1.0] * len(documents)


class Store:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.memories = [
            SimpleNamespace(
                id=uuid4(),
                raw_content="workspace memory",
                workspace="camillo",
                base_importance=0.8,
                access_count=0,
                last_accessed_at=now,
            ),
            SimpleNamespace(
                id=uuid4(),
                raw_content="other memory",
                workspace="other",
                base_importance=0.8,
                access_count=0,
                last_accessed_at=now,
            ),
        ]
        self.accessed: list = []

    async def vector_candidates(self, _embedding: list[float], _limit: int):
        return [(memory, 1.0) for memory in self.memories]

    async def full_text_search_candidates(self, _query: str, _limit: int):
        return [(memory, 1.0) for memory in self.memories]

    async def mark_accessed(self, ids):
        self.accessed.extend(ids)


@pytest.mark.asyncio
async def test_workspace_is_a_ranking_hint_and_public_recall_mutates_access() -> None:
    store = Store()
    service = RecallService(store, Provider())
    results = await service.recall("memory", 2, "camillo")
    assert results[0].memory.workspace == "camillo"
    assert len(store.accessed) == 2


@pytest.mark.asyncio
async def test_read_only_search_does_not_mutate_access() -> None:
    store = Store()
    service = RecallService(store, Provider())
    await service.search("memory", 2, "missing")
    assert store.accessed == []
