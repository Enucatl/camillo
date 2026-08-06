from pathlib import Path

from camillo import worker
from camillo.worker import build_parser


def test_worker_parser_accepts_once_and_loop_modes() -> None:
    """Ensure worker CLI keeps documented execution modes importable."""
    parser = build_parser()

    once = parser.parse_args(["--once", "--namespace", "repo:backend"])
    loop = parser.parse_args(["--loop", "--namespace", "repo:backend", "--dry-run"])

    assert once.once is True
    assert once.loop is False
    assert once.namespace == "repo:backend"
    assert loop.loop is True
    assert loop.dry_run is True


def test_write_heartbeat_touches_configured_path(monkeypatch, tmp_path: Path) -> None:
    """Ensure the worker exposes fresh loop activity to its healthcheck."""
    heartbeat = tmp_path / "heartbeat"
    monkeypatch.setattr(worker, "WORKER_HEARTBEAT_PATH", heartbeat)

    worker.write_heartbeat()

    assert heartbeat.is_file()
