## Test Commands

Run the fast local pytest suite during normal development:

```bash
uv run pytest
```

Run the full containerized test stack before changing Docker, Compose,
migrations, database behavior, or deployment wiring:

```bash
docker compose -f docker-compose.test.yml up --build --force-recreate --exit-code-from pytest pytest
docker compose -f docker-compose.test.yml down --remove-orphans
```

The test stack builds the application image, starts an isolated tmpfs-backed
PostgreSQL/pgvector database, runs Alembic migrations, waits for the app
healthcheck, and runs pytest with PostgreSQL integration tests enabled.

Run performance tests only when explicitly needed:

```bash
RUN_PERF_TESTS=1 uv run pytest -m performance
```

## Formatting

Before committing Python changes, run:

```bash
uv run ruff format .
```
