import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("APP_NAME", "Camillo")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("EMBEDDING_DIM", "1024")
os.environ.setdefault("LITELLM_COMPLETION_MODEL", "openrouter/google/gemma-4-31b-it:free")
os.environ.setdefault("LITELLM_EMBEDDING_MODEL", "openrouter/baai/bge-m3")
os.environ.setdefault("LITELLM_RERANK_MODEL", "openrouter/cohere/rerank-4-pro")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("PHOENIX_TRACING_ENABLED", "false")
os.environ.setdefault(
    "PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix-otlp.docker.home.arpa/v1/traces"
)
os.environ.setdefault("PHOENIX_PROJECT_NAME", "camillo-test")
os.environ.setdefault("DECAY_RATE", "0.01")
os.environ.setdefault("RECALL_TOP_K", "5")
os.environ.setdefault("RECALL_VECTOR_LIMIT", "30")
os.environ.setdefault("RECALL_FULL_TEXT_SEARCH_LIMIT", "30")
os.environ.setdefault("HEBBIAN_EDGE_THRESHOLD", "2.0")
