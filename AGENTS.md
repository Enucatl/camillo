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

## Code Documentation

Every new Python function and class must include a Google-style docstring. The
docstring should explain why the code exists and the design constraint it
protects, not just restate what the code does. Include `Args:` and `Returns:`
sections when they clarify the contract.

Add inline comments only where they explain non-obvious tradeoffs, invariants, or
provider/database quirks that future maintainers would otherwise need to infer.

## Memory Semantics

MCP `recall_memory` is intentionally adaptive, not read-only. Calling it should
refresh memory access counts and reinforce Hebbian co-recall edges so useful
memories stay active and associations form through use. Do not expose access
counts or graph mutation details in the MCP recall response unless explicitly
requested; clients need recalled content and score provenance, not bookkeeping.

Use the internal read-only recall path only for backend policy checks such as
reconciliation, duplicate detection, diagnostics, or other flows where looking
up related memories must not train the graph.
