from fastapi import FastAPI

from cognitive_memory.api import routes_health, routes_ingest, routes_recall
from cognitive_memory.logging_config import configure_logging
from cognitive_memory.settings import settings
from cognitive_memory.tracing_config import configure_phoenix_tracing

configure_logging()
configure_phoenix_tracing()

app = FastAPI(title=settings.app_name)
app.include_router(routes_health.router)
app.include_router(routes_ingest.router)
app.include_router(routes_recall.router)
