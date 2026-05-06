from uuid import UUID, uuid4

import pytest

from camillo.db.models import DreamRun
from camillo.stores.dream_store import DreamStore


class FakeSession:
    """Minimal async session for DreamStore unit tests."""

    def __init__(self):
        """Start with no rows and no flushes."""
        self.rows: dict[UUID, DreamRun] = {}
        self.flush_count = 0

    def add(self, row: DreamRun) -> None:
        """Capture inserted model rows."""
        self.rows[row.id] = row

    async def flush(self) -> None:
        """Track that stores flush but do not commit."""
        self.flush_count += 1

    async def get(self, model: type[DreamRun], row_id: UUID) -> DreamRun | None:
        """Return captured rows by primary key."""
        return self.rows.get(row_id)


@pytest.mark.asyncio
async def test_dream_store_creates_completes_and_fails_runs() -> None:
    """Protect dream run lifecycle bookkeeping."""
    session = FakeSession()
    store = DreamStore(session)  # type: ignore[arg-type]

    run = await store.create_run("repo", dry_run=False, metadata={"seed_limit": 5})
    source_id = uuid4()
    created_id = uuid4()
    completed = await store.complete_run(
        run.id,
        seed_memory_ids=[source_id],
        source_memory_ids=[source_id],
        created_memory_ids=[created_id],
        clusters_considered=1,
        clusters_dreamed=1,
        memories_created=1,
        dry_run=False,
        metadata={"completed": True},
    )
    failed = await store.fail_run(run.id, "boom")

    assert completed.id == run.id
    assert completed.created_memory_ids == [created_id]
    assert completed.metadata_json["seed_limit"] == 5
    assert completed.metadata_json["completed"] is True
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert session.flush_count == 3
