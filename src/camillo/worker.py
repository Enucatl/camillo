import argparse
import asyncio
from pathlib import Path

from loguru import logger

from camillo.ai.llm_service import LiteLLMService
from camillo.cognitive.dreaming_service import DreamingService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import AsyncSessionLocal
from camillo.settings import settings
from camillo.stores.dream_store import DreamStore
from camillo.stores.graph_store import GraphStore
from camillo.stores.memory_store import MemoryStore
from camillo.stores.relation_store import RelationStore

WORKER_HEARTBEAT_PATH = Path("/tmp/dreaming-worker-heartbeat")


def write_heartbeat() -> None:
    """Record that the worker loop is alive for the container healthcheck."""
    WORKER_HEARTBEAT_PATH.touch()


def build_parser() -> argparse.ArgumentParser:
    """Build the dreaming worker CLI parser.

    Returns:
        Parser accepting one-shot and loop execution modes.
    """
    parser = argparse.ArgumentParser(description="Run Camillo dreaming consolidation.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one dreaming pass and exit.")
    mode.add_argument("--loop", action="store_true", help="Run dreaming repeatedly.")
    parser.add_argument("--namespace", default=None, help="Namespace to consolidate.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write consolidation effects."
    )
    parser.add_argument("--interval-seconds", type=int, default=None, help="Loop sleep interval.")
    return parser


async def run_once(namespace: str, *, dry_run: bool | None = None) -> None:
    """Run one transaction-scoped dreaming pass.

    Args:
        namespace: Memory partition to consolidate.
        dry_run: Optional override for dry-run behavior.
    """
    if not settings.dreaming_enabled:
        logger.info("Dreaming is disabled; skipping run for namespace {}", namespace)
        return

    async with AsyncSessionLocal() as db:
        try:
            memory_store = MemoryStore(db)
            graph_store = GraphStore(db)
            relation_store = RelationStore(db)
            dream_store = DreamStore(db)
            llm_service = LiteLLMService()
            recall_service = RecallService(memory_store, graph_store, llm_service)
            reconciliation_service = MemoryReconciliationService(
                memory_store,
                relation_store,
                recall_service,
                llm_service,
            )
            service = DreamingService(
                memory_store,
                graph_store,
                relation_store,
                dream_store,
                reconciliation_service,
                llm_service,
            )
            report = await service.run_once(namespace, dry_run=dry_run)
            await db.commit()
            logger.info(
                "Dreaming run {} status={} clusters_dreamed={} memories_created={}",
                report.dream_run_id,
                report.status,
                report.clusters_dreamed,
                report.memories_created,
            )
        except Exception:
            await db.rollback()
            logger.exception("Dreaming run failed")
            raise


async def main_async(argv: list[str] | None = None) -> None:
    """Run the worker from parsed CLI arguments.

    Args:
        argv: Optional argument list for tests.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    namespace = args.namespace or settings.dreaming_namespace
    dry_run = True if args.dry_run else None
    interval = args.interval_seconds or settings.dreaming_interval_seconds

    if not args.loop:
        await run_once(namespace, dry_run=dry_run)
        return

    if settings.dreaming_run_on_start:
        await run_once(namespace, dry_run=dry_run)

    while True:
        write_heartbeat()
        await asyncio.sleep(interval)
        await run_once(namespace, dry_run=dry_run)


def main() -> None:
    """Synchronous entrypoint for `python -m camillo.worker`."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
