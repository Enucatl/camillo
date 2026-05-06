from datetime import UTC, datetime
from uuid import uuid4

import pytest

from camillo.api import routes_dreaming
from camillo.schemas.dreaming import DreamRequest, DreamRunReport


class FakeDreamingService:
    """Replace dreaming internals so the route test isolates response shape."""

    def __init__(self, *args):
        """Accept production constructor arguments."""

    async def run_once(
        self,
        namespace: str,
        *,
        seed_limit: int | None = None,
        dry_run: bool | None = None,
    ) -> DreamRunReport:
        """Return a stable dream report."""
        return DreamRunReport(
            dream_run_id=uuid4(),
            namespace=namespace,
            status="dry_run" if dry_run else "completed",
            started_at=datetime.now(UTC),
            clusters_considered=0,
            clusters_dreamed=0,
            memories_created=0,
            dry_run=bool(dry_run),
            clusters=[],
        )


class FakeSession:
    """Minimal async session surface used by the route."""

    def __init__(self):
        """Track transaction calls."""
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        """Mark that the route committed after service success."""
        self.committed = True

    async def rollback(self) -> None:
        """Mark rollback when route failures occur."""
        self.rolled_back = True


@pytest.mark.asyncio
async def test_dream_route_returns_report_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `/dream` returns the DreamRunReport model."""
    monkeypatch.setattr(routes_dreaming, "DreamingService", FakeDreamingService)
    db = FakeSession()
    request = DreamRequest(namespace="repo", seed_limit=5, dry_run=True)

    report = await routes_dreaming.dream(request, db=db)

    assert report.namespace == "repo"
    assert report.status == "dry_run"
    assert report.dry_run is True
    assert db.committed is True
