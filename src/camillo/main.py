from fastapi import FastAPI

from camillo.api import routes_health, routes_ingest, routes_recall
from camillo.logging_config import configure_logging
from camillo.settings import settings
from camillo.tracing_config import configure_phoenix_tracing

configure_logging()
configure_phoenix_tracing()

app = FastAPI(title=settings.app_name)
app.include_router(routes_health.router)
app.include_router(routes_ingest.router)
app.include_router(routes_recall.router)
