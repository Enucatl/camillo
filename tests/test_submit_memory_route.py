from uuid import uuid4

import pytest

from camillo.api import routes_submit_memory
from camillo.schemas.submit_memory import MemorySubmissionReport, SubmitMemoryRequest


class FakeReconciliationService:
    """Replace service internals so the route test isolates API shape."""

    def __init__(self, *args):
        """Accept production constructor arguments."""

    async def submit_memory(self, **kwargs: object) -> MemorySubmissionReport:
        """Return a stable report while preserving request forwarding."""
        return MemorySubmissionReport(
            outcome="created",
            created_memory_id=uuid4(),
            message=f"stored {kwargs['namespace']}",
        )


class FakeSession:
    """Minimal async session surface used by the route."""

    def __init__(self):
        """Track commit calls for the route assertion."""
        self.committed = False

    async def commit(self) -> None:
        """Mark that the route committed after service success."""
        self.committed = True


@pytest.mark.asyncio
async def test_submit_memory_route_returns_expected_report_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure `/submit_memory` serializes the reconciliation report model."""
    monkeypatch.setattr(
        routes_submit_memory, "MemoryReconciliationService", FakeReconciliationService
    )
    db = FakeSession()
    request = SubmitMemoryRequest(namespace="repo", content="Use pgvector.")

    report = await routes_submit_memory.submit_memory(request, db=db)

    assert report.outcome == "created"
    assert report.created_memory_id is not None
    assert report.message == "stored repo"
    assert db.committed is True
