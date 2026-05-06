# Camillo

Camillo is a Python 3.14 cognitive memory service. It stores conversation
episodes, durable facts, preferences, procedures, and profile-like context in
PostgreSQL with pgvector, then exposes recall and memory mutation through both a
FastAPI HTTP API and a Streamable HTTP MCP server.

The implemented design is intentionally Postgres-native and modular:

- PostgreSQL stores memory rows, vector embeddings, lifecycle fields, Hebbian
  associative edges, semantic memory relations, and dreaming run audit records.
- FastAPI owns the public HTTP routes and mounts the MCP application at `/mcp`.
- Cognitive services orchestrate ingestion, recall, reconciliation, graph
  reinforcement, and dreaming without owning provider or database details.
- LiteLLM is the default provider adapter for completion, embeddings, reranking,
  relationship classification, and dream synthesis.
- Docker Compose runs the app as `teatro`, migrations as `sipario`, PostgreSQL
  as `postgres`, and background consolidation as `dreaming_worker`.

Use stable namespaces such as `repo:<repo_name>`, `user:<id>`, or
`workspace:<id>` to isolate memory. `camillo` is the service name, not the
default memory namespace.

## Current Architecture

```text
Agent / IDE / HTTP client
  |
  |-- HTTP: /health, /ingest, /recall, /submit_memory, /dream
  |-- MCP:  recall_memory, record_interaction, submit_memory, memory_stats
  v
FastAPI app in teatro
  |
  |-- API route adapters
  |-- mounted FastMCP Streamable HTTP app at /mcp
  v
Cognitive service layer
  |
  |-- IngestionService
  |-- RecallService
  |-- MemoryReconciliationService
  |-- DreamingService
  |-- reinforcement and scope policy utilities
  v
Ports and provider adapters
  |
  |-- EmbeddingProvider
  |-- Reranker
  |-- LiteLLMService
  v
Postgres storage adapters
  |
  |-- MemoryStore
  |-- GraphStore
  |-- RelationStore
  |-- DreamStore
  v
PostgreSQL 18 + pgvector + pg_trgm
```

The main architectural choice is to treat memory as several composable
pipelines, not as a single RAG function. Retrieval, ranking, lifecycle policy,
semantic relations, associative graph behavior, and consolidation can change
independently because the services depend on ports rather than concrete
provider or database code.

## Runtime Services

Compose services:

- `postgres`: PostgreSQL 18 with pgvector, `pg_trgm`, and persistent storage.
- `sipario`: one-shot Alembic migration runner.
- `teatro`: FastAPI application server and mounted MCP HTTP endpoint.
- `dreaming_worker`: looped background worker for episodic-to-durable
  consolidation.

The Compose stack extends the local security baseline in
`../compose-security-baseline/hardening.yml`, keeps the application on an
internal network, exposes `teatro` through Traefik, and reads the database
password from `secrets/postgres_password`.

## Data Model

Camillo currently uses four main persisted concepts:

- `memories`: namespace-scoped memory records with raw content, embedding,
  memory type, lifecycle status, scope, confidence, source metadata,
  importance, access count, timestamps, and arbitrary JSON metadata.
- `hebbian_edges`: undirected associative edges between memories. These model
  adjacency, co-access, and repeated co-recall. They are weight-based and
  separate from semantic meaning.
- `memory_relations`: directed semantic or lifecycle relations such as
  `extends`, `supersedes`, `contradicts`, `refines`, `exception_to`, and
  `consolidates`.
- `dream_runs`: audit rows for consolidation passes, including seed IDs, source
  IDs, created memory IDs, counts, status, timing, and error metadata.

Memory lifecycle is explicit. Normal recall only returns active memories.
Reconciliation may mark older memories as `deprecated` or `superseded`.
Dreaming marks successfully promoted source episodes as `consolidated` so they
are not repeatedly promoted.

Memory types include:

- `episodic`: raw user/assistant interaction turns.
- `semantic`: durable facts or project knowledge.
- `preference`: user or project preferences.
- `procedural`: reusable procedures or operating rules.
- `relationship`: relationship facts.
- `profile`: profile-like facts.
- `core`: high-priority durable constraints.

Memory scope controls cross-namespace reuse:

- `local`: only the memory namespace.
- `shared`: eligible for recall from other namespaces when shared recall is
  enabled.
- `global`: broadly reusable memory.

## Ingestion Pipeline

`POST /ingest` and the MCP `record_interaction` tool store raw interaction
history as episodic memory.

```text
user_msg + ai_msg
  -> deterministic rule-based importance scoring
  -> embedding through LiteLLM
  -> insert episodic memory
  -> find previous active memory in the same session
  -> create or strengthen a Hebbian adjacency edge
```

The local importance score becomes the memory's base importance. A `session_id`
is optional, but using one lets Camillo connect adjacent turns into the
associative graph.

## Recall Pipeline

`POST /recall` and the MCP `recall_memory` tool run the full cognitive recall
path.

```text
query
  -> embed query
  -> vector candidates from pgvector
  -> lexical candidates from pg_trgm similarity
  -> reciprocal rank fusion
  -> optional LiteLLM reranking
  -> relevance threshold
  -> ACT-R inspired activation scoring
  -> namespace/scope affinity scoring
  -> weighted final score
  -> diversity filter
  -> top K primary memories
  -> optional Hebbian neighbor expansion
  -> access bookkeeping and clique reinforcement
```

Primary memories are direct matches from the hybrid retrieval pipeline.
Hebbian memories are appended afterward as graph context. They enrich recall but
do not displace the primary ranked results.

Recall responses include score provenance:

- `vector_score`
- `text_score`
- `rrf_score`
- `rerank_score`
- `activation_score`
- `scope_affinity_score`
- `final_score`

HTTP recall has side effects by design: returned memories get their access count
and last-access timestamp updated, and co-returned memories reinforce a Hebbian
clique. Internal policy checks and MCP recall use the read-only recall path so
duplicate detection and context lookup do not mutate memory. Disable
`include_hebbian` for strict direct retrieval, or `include_shared` for strict
namespace-local recall.

## Durable Memory Reconciliation

`POST /submit_memory` and the MCP `submit_memory` tool are the policy boundary
for memories intended to affect future behavior beyond the current episode.

Inputs:

- `namespace`
- `content`
- `intent`: `auto`, `remember`, `correct`, or `forget`
- `memory_type`: `semantic`, `preference`, `procedural`, `relationship`,
  `profile`, or `core`
- `scope`: optional `local`, `shared`, or `global`
- `evidence`: optional source text or rationale
- `confidence`: optional `0.0` to `1.0`

Implemented reconciliation flow:

```text
durable memory candidate
  -> normalize type, scope, and confidence
  -> recall related active memories without Hebbian expansion
  -> classify relationship with LiteLLM
  -> reinforce duplicates or confirmations
  -> deprecate explicit forget targets
  -> create a new durable memory when needed
  -> create semantic relation rows for meaningful relationships
  -> supersede or deprecate older memories when policy allows
  -> return a transparent report
```

This intentionally replaces lower-level operations like `force_remember` or
manual status updates. Clients express intent; Camillo owns duplicate handling,
contradiction handling, lifecycle transitions, relation tracking, and metadata.

## Dreaming Consolidation

Dreaming is Camillo's background consolidation pass. It promotes useful clusters
of graph-connected episodic memories into durable memory candidates.

```text
active episodic seed selection
  -> ACT-R activation threshold
  -> Hebbian cluster traversal
  -> cluster size, age, and importance validation
  -> LiteLLM dream synthesis
  -> proposal validation
  -> submit proposals through reconciliation
  -> create consolidation relations to source episodes
  -> mark source episodes consolidated after successful promotion
  -> record dream run audit details
```

Dreaming can run through:

- `dreaming_worker` in Docker Compose.
- `python -m camillo.worker --once`.
- `python -m camillo.worker --loop`.
- `POST /dream` for an admin-triggered pass.

Use `DREAMING_DRY_RUN=true` or the request/CLI dry-run option to inspect
proposals without writing consolidation effects.

## MCP Tool Surface

The MCP server is mounted under the FastAPI app at `/mcp` using Streamable HTTP.
GET probes to `/mcp` return `405` with `Allow: POST`; clients should connect
with the Streamable HTTP protocol.

Exposed tools:

- `recall_memory`: read active memories through the recall pipeline.
- `record_interaction`: write one raw user/assistant turn as episodic memory.
- `submit_memory`: reconcile a durable memory candidate.
- `memory_stats`: return counts by type and lifecycle status for one namespace.

The MCP layer is deliberately simple. It does not expose raw graph writes,
manual lifecycle mutation, or low-level storage operations.

## HTTP API

### Health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Ingest Interaction

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

### Recall Memory

```bash
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "query": "What database did we choose?",
    "top_k": 5,
    "include_hebbian": true,
    "include_shared": true
  }'
```

### Submit Durable Memory

```bash
curl -X POST http://localhost:8000/submit_memory \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "content": "The backend memory service uses PostgreSQL with pgvector.",
    "intent": "remember",
    "memory_type": "semantic",
    "confidence": 0.9
  }'
```

### Run Dreaming Once

```bash
curl -X POST http://localhost:8000/dream \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "seed_limit": 5,
    "dry_run": true
  }'
```

## Local Run

Create the environment and secret:

```bash
cp .env.example .env
mkdir -p secrets
printf 'change-me\n' > secrets/postgres_password
docker compose up --build
```

Development defaults route LiteLLM through OpenRouter:

- completion: `openrouter/google/gemma-4-31b-it:free`
- embedding: `openrouter/baai/bge-m3`
- rerank: `openrouter/cohere/rerank-4-pro`

Set `OPENROUTER_API_KEY` in `.env` before calling routes or MCP tools that need
LLM providers. `/health` does not require an LLM key.

For local non-Compose runs, either provide `DATABASE_URL` or set the Postgres
parts and a readable `POSTGRES_PASSWORD_FILE`.

## Key Configuration

Provider settings:

- `LITELLM_COMPLETION_MODEL`
- `LITELLM_EMBEDDING_MODEL`
- `LITELLM_RERANK_MODEL`
- `OPENROUTER_API_KEY`
- `EMBEDDING_DIM`

Recall settings:

- `RECALL_TOP_K`
- `RECALL_VECTOR_LIMIT`
- `RECALL_FULL_TEXT_SEARCH_LIMIT`
- `RERANK_ENABLED`
- `RERANK_MIN_SCORE`
- `RRF_K`
- `RECALL_CANDIDATE_LIMIT`
- `DIVERSITY_ENABLED`
- `DIVERSITY_SIMILARITY_THRESHOLD`
- `HEBBIAN_SPREAD_ENABLED`
- `HEBBIAN_SPREAD_LIMIT`
- `HEBBIAN_EDGE_THRESHOLD`
- `REINFORCEMENT_ENABLED`
- `REINFORCEMENT_EDGE_INCREMENT`
- `DECAY_RATE`

Dreaming settings:

- `DREAMING_ENABLED`
- `DREAMING_INTERVAL_SECONDS`
- `DREAMING_RUN_ON_START`
- `DREAMING_NAMESPACE`
- `DREAMING_DRY_RUN`
- `DREAMING_SEED_LIMIT`
- `DREAMING_CLUSTER_MAX_SIZE`
- `DREAMING_CLUSTER_MIN_SIZE`
- `DREAMING_MAX_DEPTH`
- `DREAMING_MIN_SEED_ACTIVATION`
- `DREAMING_MIN_EDGE_WEIGHT`
- `DREAMING_MAX_CLUSTER_AGE_DAYS`
- `DREAMING_MIN_CLUSTER_TOTAL_IMPORTANCE`
- `DREAMING_MIN_SYNTHESIS_CONFIDENCE`
- `DREAMING_MAX_MEMORIES_PER_CLUSTER`
- `DREAMING_SOURCE_PENALTY`
- `DREAMING_MIN_SOURCE_IMPORTANCE`
- `DREAMING_RELATION_CONFIDENCE`
- `DREAMING_MODEL`

MCP deployment setting:

- `MCP_ALLOWED_HOSTS`

## Development

Run formatting and the fast local test suite:

```bash
uv run ruff format .
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
healthcheck, and executes pytest with PostgreSQL integration tests enabled.

Run the real PostgreSQL/pgvector integration tests against a migrated database:

```bash
RUN_DB_TESTS=1 \
POSTGRES_USER=camillo \
POSTGRES_PASSWORD_FILE=secrets/postgres_password \
POSTGRES_DB=camillo \
POSTGRES_HOST=localhost \
POSTGRES_PORT=5432 \
uv run pytest tests/test_postgres_memory_flow.py
```

Run synthetic performance tests only when needed:

```bash
RUN_PERF_TESTS=1 uv run pytest -m performance
```

Run the Scalene performance helper:

```bash
scripts/run_scalene_profile.sh
```

The report is written to `reports/scalene-performance.txt`.

## Phoenix Tracing

LLM tracing is optional. To send LiteLLM spans to a self-hosted Phoenix instance:

```bash
PHOENIX_TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=https://phoenix-otlp.${DOCKER_DOMAIN}/v1/traces
PHOENIX_PROJECT_NAME=camillo
```

The Docker image installs the tracing extra. The `teatro` service mounts
`secrets/fullchain.pem` and `secrets/key.pem` as the OTLP client certificate and
key, matching the mTLS setup used by the surrounding Docker environment.

## Implemented Boundary

The current implementation covers the original foundation, full recall, MCP,
memory reconciliation, scoped memory sharing, and dreaming consolidation plans.
It does not implement a filesystem watcher, non-Postgres graph backend, local
embedding service, automatic secret redaction, or administrative bulk memory
management. Those remain outside the current code boundary.
