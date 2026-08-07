# Camillo

Camillo is a Python memory backbone for agentic AI. It stores one personal
memory corpus in PostgreSQL with pgvector and optional workspace affinity.

All conversation turns are automatically captured as redacted `episode`
memories through `POST /ingest`. Agents query the corpus with hybrid vector and
lexical retrieval. Ranking is fixed at 75% relevance, 15% retrieval activation,
and 10% exact workspace match. Workspace is only a tie-breaking hint; an
incorrect or missing workspace never hides a memory.

Durable memory types are `episode`, `fact`, `preference`, and `procedure`.
`remember_memory` defaults to `fact`, detects exact and semantic duplicates,
and reinforces existing rows. `replace_memory` and `forget_memory` require one
explicit active memory ID. Retrieval decay affects ranking only: inactive rows
remain stored until explicitly forgotten, replaced, or consolidated.

The service has no namespace/scope controls, Hebbian graph, or semantic
relation table. Conversation `session_id`, embeddings, replacement lineage,
and `dream_runs` remain. Dreaming is a scheduled one-shot command:

```bash
docker compose run --rm dreaming_worker python -m camillo.worker --once
```

The default Compose stack does not start a resident dreaming worker. Schedule
the command daily with the host's systemd timer or cron and apply a container
memory limit in that scheduler. The worker skips provider calls when fewer than
two qualifying episodes exist and exits after one run.

## API

- `POST /ingest`: `user_msg`, `ai_msg`, optional `session_id` and `workspace`.
- `POST /recall`: `query`, optional `top_k` and `workspace`.
- `POST /remember_memory`: durable content, type, evidence, workspace.
- `POST /replace_memory`: explicit `memory_id` and replacement content.
- `POST /forget_memory`: explicit `memory_id` and optional reason.
- `POST /dream`: manually trigger one one-shot consolidation pass.
- MCP exposes `recall_memory`, `remember_memory`, `replace_memory`,
  `forget_memory`, and `memory_stats` with the same contracts.

## Development

```bash
uv run ruff format .
uv run pytest
docker compose -f docker-compose.test.yml up --build --force-recreate --exit-code-from pytest pytest
docker compose -f docker-compose.test.yml down --remove-orphans
```

Apply schema changes with `alembic upgrade head`. Migration `0006` maps legacy
namespaces and memory types, preserves memory fields and replacement links,
then removes obsolete graph/relation tables and namespace-specific dreaming
fields. Snapshot the database before production migration; deleted graph and
relation data cannot be reconstructed on downgrade.
