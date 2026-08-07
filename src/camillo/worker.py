import argparse
import asyncio

from loguru import logger

from camillo.ai.llm_service import get_inference_service
from camillo.cognitive.dreaming_service import DreamingService
from camillo.cognitive.recall_service import RecallService
from camillo.cognitive.reconciliation_service import MemoryReconciliationService
from camillo.db.session import AsyncSessionLocal
from camillo.settings import settings
from camillo.stores.dream_store import DreamStore
from camillo.stores.memory_store import MemoryStore


def build_parser() -> argparse.ArgumentParser:
    """Build the one-shot dreaming command-line interface."""
    parser = argparse.ArgumentParser(description="Run one Camillo dreaming pass.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Avoid promotion writes.")
    return parser


async def run_once(*, dry_run: bool | None = None) -> None:
    """Run exactly one transaction-scoped dreaming pass and exit."""
    if not settings.dreaming_enabled:
        return
    async with AsyncSessionLocal() as db:
        store = MemoryStore(db)
        provider = get_inference_service()
        service = DreamingService(
            store,
            DreamStore(db),
            MemoryReconciliationService(store, RecallService(store, provider), provider),
            provider,
        )
        report = await service.run_once(dry_run=dry_run)
        await db.commit()
        logger.info(
            "Dreaming run {} created {} memories", report.dream_run_id, report.memories_created
        )


async def main_async(argv: list[str] | None = None) -> None:
    """Parse arguments and execute one run; no loop is supported."""
    args = build_parser().parse_args(argv)
    try:
        await run_once(dry_run=True if args.dry_run else None)
    finally:
        await get_inference_service().close()


def main() -> None:
    """Run the one-shot worker entrypoint."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
