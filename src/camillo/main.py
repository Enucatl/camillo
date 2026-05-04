from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from camillo.api import routes_health, routes_ingest, routes_recall, routes_submit_memory
from camillo.logging_config import configure_logging
from camillo.mcp_server.server import mcp
from camillo.settings import settings
from camillo.startup_checks import warn_missing_provider_keys
from camillo.tracing_config import configure_phoenix_tracing

configure_logging()
warn_missing_provider_keys()
configure_phoenix_tracing()

mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start mounted MCP lifecycle tasks with the main ASGI process.

    Starlette does not run mounted sub-application lifespans automatically, so
    the combined API/MCP container must enter FastMCP's session manager here.
    """
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(routes_health.router)
app.include_router(routes_ingest.router)
app.include_router(routes_recall.router)
app.include_router(routes_submit_memory.router)
app.mount("/mcp", mcp_app)
