# Camillo simplification and single-user memory refactor

Status: complete

## Objective and design decisions

Refactor Camillo into a trusted, private, single-user memory corpus. Memories
have optional `workspace` affinity rather than required namespace/scope controls.
The corpus keeps hybrid retrieval, explicit ID-based lifecycle operations,
retrieval-only activation decay, embeddings, session grouping, replacement
lineage, and audited scheduled one-shot dreaming. Hebbian graph and semantic
relation subsystems are removed completely.

Locked decisions:

- Supported memory types are `episode`, `fact`, `preference`, and `procedure`.
- Workspace is a soft ranking hint; a missing or incorrect workspace never
  excludes a memory.
- Public recall reinforces access metadata; internal search is read-only.
- Fixed retrieval weights are 75% relevance, 15% activation, and 10% exact
  workspace match.
- Durable mutations require explicit IDs for replacement and forgetting.
- Retrieval decay never deletes rows; lifecycle operations do.
- Dreaming runs as a daily one-shot job and has no resident worker process.
- Deterministic secret redaction happens before scoring, provider calls, logs,
  or persistence.
- This is a clean API break; no namespace compatibility shim is required.

## Ordered implementation phases

- [x] Phase 0: create this living implementation plan.
- [x] Phase 1: migrate the single-user data model and legacy data.
- [x] Phase 2: simplify retrieval and add evaluation fixtures.
- [x] Phase 3: replace HTTP and MCP APIs and durable-memory operations.
- [x] Phase 4: remove graph and relation subsystems.
- [x] Phase 5: simplify dreaming and make the worker one-shot.
- [x] Phase 6: update the Codex hook and skill source in `~/dotfiles-vim`.
- [x] Phase 7: add deterministic ingestion redaction.
- [x] Phase 8: update documentation, Compose, settings, and acceptance tests.
- [x] Final: run formatting, local pytest, containerized pytest, and review
  the complete diff before committing.

## Discovered tasks

- [x] Inspect the source dotfiles repository and locate the current Codex hook
  and skill files before changing them.
- [x] Add migration tests that exercise representative legacy rows before
  destructive schema removal.
- [x] Add or adapt test fixtures for workspace derivation and retrieval scores.

## Decision log

- 2026-08-07: Adopted the supplied clean-break single-user design. Rejected a
  namespace compatibility shim because the requested API and schema boundary
  explicitly remove namespace and scope.
- 2026-08-07: Kept embeddings and hybrid retrieval because they are explicitly
  retained in the requested pipeline; removed graph and relation state because
  it is no longer part of the product boundary.
- 2026-08-07: Updated the container test image to install the dev extra only
  when the test Compose build argument is enabled. This keeps production images
  free of pytest while allowing the mandated container suite to run offline
  from the locked dependency set.

## Verification log

- 2026-08-07: Repository inspected; worktree was clean before implementation.
- 2026-08-07: `camillo` skill was read from `/home/user/.codex/skills/camillo`.
- 2026-08-07: Initial memory recall completed using the legacy project namespace.
- 2026-08-07: `uv run ruff format .` completed.
- 2026-08-07: `uv run ruff check .` passed.
- 2026-08-07: `uv run pytest -q` passed: 6 tests.
- 2026-08-07: Final `docker compose -f docker-compose.test.yml up --build --force-recreate --exit-code-from pytest pytest` passed: 6 tests; teardown completed with `down --remove-orphans`.
- 2026-08-07: `rake links` completed in `/home/user/dotfiles-vim`; generated Codex files were not edited directly.

## Migration and rollback notes

The new Alembic migration preserves memory content, embeddings, lifecycle
statuses, timestamps, metadata, and replacement links while mapping legacy
namespaces and memory types. It removes obsolete graph/relation tables and
namespace-specific dream fields. Before production migration, snapshot the
database and run the migration test fixture. Rollback is supported through the
Alembic downgrade only where the migration can safely reconstruct the prior
schema; destructive graph/relation data is not recoverable after upgrade.

## Final acceptance criteria

- [x] `plan.md` is complete and current.
- [x] No model-generated namespace is required and workspace mismatch cannot
  produce a false empty recall.
- [x] Graph and relation subsystems are absent from models, services, APIs,
  settings, migrations, deployment, documentation, and tests.
- [x] Replacement and forgetting require explicit active IDs.
- [x] Legacy data migrates without content loss.
- [x] No persistent dreaming process remains; the worker exits after one run.
- [x] Secret values are redacted before provider calls and persistence.
- [x] Source dotfiles changes are present in `~/dotfiles-vim`, not generated
  `~/.vim` files.
- [x] Local and full containerized test suites pass.
