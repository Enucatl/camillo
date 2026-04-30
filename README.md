# Camillo

A Python 3.14 Postgres-native cognitive memory service with FastAPI, async SQLAlchemy,
pgvector, LiteLLM provider wrappers, ACT-R inspired activation, and Hebbian edge
reinforcement. The main engine service is `teatro`.

Development defaults route LiteLLM through OpenRouter:

- completion: `openrouter/google/gemma-4-31b-it:free`
- embedding: `openrouter/baai/bge-m3`
- rerank: `openrouter/cohere/rerank-4-pro`

Set `OPENROUTER_API_KEY` in `.env` before calling `/ingest` or `/recall` with
the default models. `/health` does not require an LLM key.

Phase 1 includes:

- `GET /health`
- `POST /ingest`
- `POST /recall`
- Alembic migrations for `memories` and `hebbian_edges`
- pgvector, trigram similarity, and basic access reinforcement
- Docker Compose security baseline hardening via `../compose-security-baseline/hardening.yml`

## Run

```bash
cp .env.example .env
docker compose up --build
```

Compose services:

- `postgres`: PostgreSQL 18 with pgvector
- `sipario`: one-shot Alembic migration runner
- `teatro`: FastAPI application server

Check health:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "session_id": "demo-session",
    "user_msg": "We decided to use Postgres with pgvector for memory.",
    "ai_msg": "I will remember that this project uses Postgres-native vector search."
  }'
```

## Recall

```bash
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "query": "What database did we choose?",
    "top_k": 5
  }'
```

## Development

```bash
uv run --extra dev ruff format .
uv run --extra dev pytest
```

Run the full containerized test stack:

```bash
docker compose -f docker-compose.test.yml up --build --force-recreate --exit-code-from pytest pytest
docker compose -f docker-compose.test.yml down --remove-orphans
```

This uses `docker-compose.test.yml` to build the application image, start an
isolated tmpfs-backed PostgreSQL/pgvector database, run Alembic migrations, wait
for the app healthcheck, and execute pytest with PostgreSQL integration tests
enabled. The stack uses its own Compose project and internal network.

Run the synthetic performance tests under Scalene:

```bash
scripts/run_scalene_profile.sh
```

The report is written to `reports/scalene-performance.txt`.

Run the real PostgreSQL/pgvector integration tests against a migrated database:

```bash
RUN_DB_TESTS=1 \
DATABASE_URL=postgresql+asyncpg://camillo:camillo@localhost:5432/camillo \
uv run --extra dev pytest tests/test_postgres_memory_flow.py
```

Add `RUN_PERF_TESTS=1 -m performance` to include the database-backed performance test.

## Phoenix Tracing

LLM tracing is optional and disabled by default. To send LiteLLM spans to a
self-hosted Phoenix instance:

```bash
PHOENIX_TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix.docker.home.arpa:6006
PHOENIX_PROJECT_NAME=camillo
```

The Docker image installs the tracing extra, so enabling these variables is
enough for the `teatro` service.
