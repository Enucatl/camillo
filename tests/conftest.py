import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("APP_NAME", "Camillo")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("EMBEDDING_DIM", "1536")
os.environ.setdefault("LITELLM_COMPLETION_MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("LITELLM_EMBEDDING_MODEL", "openai/text-embedding-3-small")
os.environ.setdefault("LITELLM_RERANK_MODEL", "")
os.environ.setdefault("PHOENIX_TRACING_ENABLED", "false")
os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.docker.home.arpa:6006")
os.environ.setdefault("PHOENIX_PROJECT_NAME", "camillo")
os.environ.setdefault("DECAY_RATE", "0.01")
os.environ.setdefault("RECALL_TOP_K", "5")
os.environ.setdefault("RECALL_VECTOR_LIMIT", "30")
os.environ.setdefault("RECALL_FTS_LIMIT", "30")
os.environ.setdefault("HEBBIAN_EDGE_THRESHOLD", "2.0")
