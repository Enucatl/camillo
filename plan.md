# Overall Architecture

The system should be built as a **modular cognitive memory service** with clean boundaries between storage, LLM routing, retrieval, graph reinforcement, consolidation, and external agent access.

Conceptually:

```text
┌─────────────────────────────────────────────────────────────┐
│                        Agent / IDE                          │
│                  MCP tools, HTTP API, watcher               │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                       API Layer                              │
│ FastAPI routes: /health, /ingest, /recall, /force_remember   │
│ MCP tools: recall_context, force_remember                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Cognitive Service Layer                   │
│ IngestionService                                              │
│ RecallService                                                 │
│ DreamingService                                               │
│ ReinforcementService                                          │
│ Security/RedactionService                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    AI Provider Layer                         │
│ LiteLLMCompletionProvider                                     │
│ LiteLLMEmbeddingProvider                                      │
│ LiteLLMReranker                                               │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     Storage Layer                            │
│ PostgresMemoryStore                                           │
│ PostgresHebbianGraphStore                                     │
│ Alembic migrations                                            │
│ pgvector + pg_trgm + FTS                                      │
└─────────────────────────────────────────────────────────────┘
```

The critical architectural decision is this:

> Treat “memory” as a set of composable pipelines, not one hard-coded RAG function.

So each pipeline should be replaceable.

## Core internal interfaces

Design around ports/adapters:

```text
EmbeddingProvider
  embed(text) -> list[float]

CompletionProvider
  score_valence(text) -> float
  synthesize_semantic_rule(cluster) -> str

Reranker
  rerank(query, documents) -> list[RerankResult]

MemoryStore
  insert_memory()
  hybrid_candidates()
  get_memories_by_ids()
  mark_accessed()
  soft_delete()

GraphStore
  link_adjacent()
  reinforce_clique()
  get_neighbors()
  decay_edges()

RecallStrategy
  recall(query, namespace) -> RecallResult

ConsolidationStrategy
  select_seeds()
  build_clusters()
  dream()
  commit_semantic_memory()
  penalize_sources()
```

Initial adapters:

```text
LiteLLMEmbeddingProvider
LiteLLMCompletionProvider
LiteLLMReranker
PostgresMemoryStore
PostgresHebbianGraphStore
DefaultRecallStrategy
DefaultDreamingStrategy
```

This lets you later swap LiteLLM reranking for a local reranker, Postgres graph traversal for Neo4j/FalkorDB, or the default decay function for a more advanced ACT-R model.

---

# Recommended Project Layout

```text
camillo/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── migrate/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── src/
│   └── camillo/
│       ├── __init__.py
│       ├── main.py
│       ├── settings.py
│       ├── logging_config.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes_health.py
│       │   ├── routes_ingest.py
│       │   └── routes_recall.py
│       │
│       ├── db/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── session.py
│       │   └── models.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── ingest.py
│       │   ├── recall.py
│       │   └── memory.py
│       │
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── llm_service.py
│       │   └── prompts.py
│       │
│       ├── cognitive/
│       │   ├── __init__.py
│       │   ├── cognitive_math.py
│       │   ├── ingestion_service.py
│       │   ├── recall_service.py
│       │   ├── reinforcement_service.py
│       │   └── dreaming_service.py
│       │
│       ├── stores/
│       │   ├── __init__.py
│       │   ├── memory_store.py
│       │   └── graph_store.py
│       │
│       ├── mcp_server/
│       │   ├── __init__.py
│       │   └── server.py
│       │
│       ├── worker.py
│       └── watcher.py
│
└── tests/
    ├── conftest.py
    ├── test_cognitive_math.py
    ├── test_ingestion_service.py
    └── test_recall_service.py
```

---

# Implementation Phases

## Phase 1 — Foundation, Docker, Database, and Minimal API

Goal: produce a runnable service with:

```text
docker compose up --build
```

and working:

```text
GET /health
POST /ingest
POST /recall
```

Phase 1 should include schema, Alembic, FastAPI, async SQLAlchemy, pgvector extension, LiteLLM embedding/valence stubs, and a minimal recall path.

No dreaming yet. No watcher yet. No MCP yet. No fancy reranking yet.

Deliverables:

```text
pyproject.toml
Dockerfile
docker-compose.yml
.env.example
settings.py
async database session
SQLAlchemy models
Alembic migration
FastAPI app
/health
/ingest
/recall
cognitive_math.py
llm_service.py
basic MemoryStore
basic GraphStore
pytest smoke tests
```

## Phase 2 — Full Recall Pipeline

Add:

```text
FTS + vector hybrid search
RRF merge
LiteLLM reranking
activation filtering
Hebbian spreading
async reinforcement
diversity de-duplication
```

At the end of Phase 2, recall should follow your exact intended sequence:

```text
hybrid candidates
→ RRF merge
→ LiteLLM rerank
→ ACT-R activation
→ top K
→ Hebbian neighbor expansion
→ reinforcement
```

## Phase 3 — MCP Server

Add MCP tools:

```text
recall_context(query, namespace)
force_remember(text, namespace, type = "semantic")
remember_interaction(user_msg, ai_msg, namespace, session_id)
forget_memory(memory_id)
memory_stats(namespace)
```

Expose either:

```text
python -m camillo.mcp_server.server
```

or run MCP beside FastAPI as a separate process in Docker.

## Phase 4 — Watcher Sidecar

Add:

```text
watcher.py
```

It should monitor a local transcript/log file, detect new user/assistant pairs, and POST them to `/ingest`.

This should be optional and configured through env vars:

```text
WATCHER_ENABLED=true
WATCHER_FILE=/data/chat.log
WATCHER_NAMESPACE=repo:backend
```

## Phase 5 — Dreaming Worker

Add:

```text
worker.py
```

It should:

```text
select highly activated episodic seeds
traverse Hebbian edges
build connected clusters
ask LiteLLM to synthesize semantic rules
store semantic memories
penalize or archive source episodic memories
```

Important: dreaming should create semantic memories without deleting raw episodic memories immediately. Let decay and lifecycle status manage that.

## Phase 6 — Security, Redaction, and Memory Lifecycle

Add:

```text
secret detection
classification labels
soft delete
archive
redaction
memory event log
```

Suggested statuses:

```text
active
archived
forgotten
redacted
deleted_soft
```

## Phase 7 — Advanced Composability

Add strategy/plugin support:

```text
ImportanceScorer
DecayModel
Reranker
GraphExpansionStrategy
ConsolidationStrategy
NamespacePolicy
MemoryTypePolicy
```

Add richer memory types:

```text
episodic
semantic
preference
relationship
procedural
profile
core
```

---

# Phase 1 — Codex-Ready Implementation Prompt

You can paste the following directly into Codex.

---

## CODEX TASK: Build Phase 1 of Camillo

Build a new Python project called `camillo`.

The goal of Phase 1 is to create a runnable FastAPI service backed by PostgreSQL 18 + pgvector, using async SQLAlchemy, Alembic migrations, LiteLLM wrappers, and a minimal ingest/recall API.

Do not implement MCP, watcher, or dreaming yet. Create clean extension points for them.

Use a `src/` project layout.

---

## 1. Create the file tree

Create this structure:

```text
.
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── migrate/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── src/
│   └── camillo/
│       ├── __init__.py
│       ├── main.py
│       ├── settings.py
│       ├── logging_config.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes_health.py
│       │   ├── routes_ingest.py
│       │   └── routes_recall.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── session.py
│       │   └── models.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── ingest.py
│       │   ├── recall.py
│       │   └── memory.py
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── llm_service.py
│       │   └── prompts.py
│       ├── cognitive/
│       │   ├── __init__.py
│       │   ├── cognitive_math.py
│       │   ├── ingestion_service.py
│       │   ├── recall_service.py
│       │   └── reinforcement_service.py
│       └── stores/
│           ├── __init__.py
│           ├── memory_store.py
│           └── graph_store.py
└── tests/
    ├── conftest.py
    ├── test_cognitive_math.py
    └── test_health.py
```

---

## 2. `pyproject.toml`

Use standard Python packaging with setuptools.

Requirements:

```toml
[project]
name = "camillo"
version = "0.1.0"
description = "A Postgres-native cognitive memory stack with LiteLLM, Hebbian edges, and ACT-R inspired decay."
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.4.0",
    "pgvector>=0.3.0",
    "litellm>=1.50.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.6.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py313"
```

---

## 3. `.env.example`

Create:

```bash
APP_NAME="Camillo"
APP_ENV=local
LOG_LEVEL=INFO

POSTGRES_USER=cognitive
POSTGRES_PASSWORD=cognitive
POSTGRES_DB=camillo
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

DATABASE_URL=postgresql+asyncpg://cognitive:cognitive@postgres:5432/camillo

EMBEDDING_DIM=1536

LITELLM_COMPLETION_MODEL=openai/gpt-4o-mini
LITELLM_EMBEDDING_MODEL=openai/text-embedding-3-small
LITELLM_RERANK_MODEL=

DECAY_RATE=0.01
RECALL_TOP_K=5
RECALL_VECTOR_LIMIT=30
RECALL_FTS_LIMIT=30
HEBBIAN_EDGE_THRESHOLD=2.0
```

---

## 4. Dockerfile

Create a simple Dockerfile:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrate ./migrate

RUN pip install -e ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "camillo.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 5. `docker-compose.yml`

Use the init-container pattern.

Important: use `pgvector/pgvector:pg18-trixie`.

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg18-trixie
    environment:
      POSTGRES_USER: cognitive
      POSTGRES_PASSWORD: cognitive
      POSTGRES_DB: camillo
    ports:
      - "5432:5432"
    volumes:
      - cognitive_pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cognitive -d camillo"]
      interval: 5s
      timeout: 5s
      retries: 20

  db_migrator:
    build: .
    env_file:
      - .env.example
    command: ["alembic", "upgrade", "head"]
    depends_on:
      postgres:
        condition: service_healthy

  cognitive_engine:
    build: .
    env_file:
      - .env.example
    ports:
      - "8000:8000"
    depends_on:
      db_migrator:
        condition: service_completed_successfully
    command: ["uvicorn", "camillo.main:app", "--host", "0.0.0.0", "--port", "8000"]

volumes:
  cognitive_pg_data:
```

---

## 6. Settings

Create `src/camillo/settings.py`.

Requirements:

Use `pydantic-settings`.

Fields:

```python
app_name: str
app_env: str
log_level: str
database_url: str
embedding_dim: int
litellm_completion_model: str
litellm_embedding_model: str
litellm_rerank_model: str | None
decay_rate: float
recall_top_k: int
recall_vector_limit: int
recall_fts_limit: int
hebbian_edge_threshold: float
```

Expose:

```python
settings = Settings()
```

Use env var names matching `.env.example`.

---

## 7. Database models

Create SQLAlchemy async-compatible models.

File: `src/camillo/db/base.py`

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

File: `src/camillo/db/models.py`

Create two models:

### `Memory`

Table name: `memories`.

Fields:

```python
id: UUID primary key
namespace: str indexed
session_id: str nullable indexed
raw_content: text
embedding: Vector(settings.embedding_dim)
type: str default "episodic"
status: str default "active"
base_importance: float default 0.5
access_count: int default 0
created_at: timezone-aware datetime
last_accessed_at: timezone-aware datetime
metadata_json: JSONB default {}
```

Use `pgvector.sqlalchemy.Vector`.

### `HebbianEdge`

Table name: `hebbian_edges`.

Fields:

```python
source_id: UUID FK memories.id primary key
target_id: UUID FK memories.id primary key
weight: float default 1.0
last_co_accessed_at: timezone-aware datetime
created_at: timezone-aware datetime
```

Add unique constraint on `(source_id, target_id)`.

Use relationships where convenient.

---

## 8. Async database session

Create `src/camillo/db/session.py`.

Expose:

```python
engine
AsyncSessionLocal
get_db()
```

Use:

```python
create_async_engine(settings.database_url, pool_pre_ping=True)
async_sessionmaker(engine, expire_on_commit=False)
```

---

## 9. Alembic

Create `alembic.ini` and `migrate/env.py`.

Alembic must support async SQLAlchemy.

`env.py` should:

```python
from camillo.db.base import Base
from camillo.db import models
from camillo.settings import settings
target_metadata = Base.metadata
```

The initial migration should create:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

Then create `memories` and `hebbian_edges`.

Indexes:

```sql
CREATE INDEX ix_memories_namespace ON memories(namespace);
CREATE INDEX ix_memories_session_id ON memories(session_id);
CREATE INDEX ix_memories_raw_content_trgm ON memories USING gin (raw_content gin_trgm_ops);
CREATE INDEX ix_memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops);
```

Because `embedding_dim` is configurable, implement the first migration for `1536` dimensions. Add a comment in the migration explaining that changing `EMBEDDING_DIM` requires a new migration.

Name the migration:

```text
0001_initial_schema.py
```

---

## 10. Schemas

Create Pydantic schemas.

### `schemas/ingest.py`

```python
class IngestRequest(BaseModel):
    namespace: str
    user_msg: str
    ai_msg: str
    session_id: str | None = None

class IngestResponse(BaseModel):
    memory_id: UUID
    namespace: str
    type: str
    base_importance: float
```

### `schemas/recall.py`

```python
class RecallRequest(BaseModel):
    namespace: str
    query: str
    top_k: int | None = None

class RecalledMemory(BaseModel):
    id: UUID
    namespace: str
    raw_content: str
    type: str
    base_importance: float
    access_count: int
    score: float

class RecallResponse(BaseModel):
    query: str
    namespace: str
    memories: list[RecalledMemory]
```

---

## 11. Cognitive math

Create `src/camillo/cognitive/cognitive_math.py`.

Implement:

```python
def calculate_activation(
    base_importance: float,
    access_count: int,
    last_accessed_at: datetime,
    *,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    ...
```

Formula:

```python
hours_since_last_access = max((now - last_accessed_at).total_seconds() / 3600, 0)
decay_score = exp(-decay_rate * hours_since_last_access)
activation = (base_importance * decay_score) + (log(access_count + 1) * 0.2)
```

Clamp result to `0.0 <= activation <= 1.5`.

Also implement:

```python
def calculate_edge_decay(
    weight: float,
    last_co_accessed_at: datetime,
    *,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    ...
```

Use exponential decay over hours.

---

## 12. LiteLLM service

Create `src/camillo/ai/llm_service.py`.

Implement class:

```python
class LiteLLMService:
    async def score_valence(self, raw_content: str) -> float:
        ...

    async def get_embedding(self, text: str) -> list[float]:
        ...

    async def rerank_results(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        ...
```

Requirements:

### `score_valence`

Use `litellm.acompletion`.

Prompt:

```text
You are scoring the long-term memory value of an AI interaction.

Return only a single float between 0.1 and 1.0.

Score high for:
- stable user preferences
- important project decisions
- architecture decisions
- emotional salience
- repeated patterns
- commitments or constraints

Score low for:
- trivial acknowledgements
- temporary chatter
- one-off small talk

Interaction:
{raw_content}
```

Parse the response defensively. If parsing fails, return `0.5`.

### `get_embedding`

Use `litellm.aembedding`.

Return the first embedding vector.

### `rerank_results`

In Phase 1, do not require real LiteLLM rerank yet. Return equal scores or a simple length-normalized placeholder. Leave a clear TODO for Phase 2.

---

## 13. Memory store

Create `src/camillo/stores/memory_store.py`.

Implement:

```python
class MemoryStore:
    def __init__(self, db: AsyncSession):
        self.db = db
```

Methods:

```python
async def insert_memory(
    namespace: str,
    raw_content: str,
    embedding: list[float],
    memory_type: str,
    base_importance: float,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Memory:
    ...
```

```python
async def get_previous_memory_in_session(
    namespace: str,
    session_id: str,
) -> Memory | None:
    ...
```

Previous memory should be latest by `created_at`.

```python
async def vector_candidates(
    namespace: str,
    embedding: list[float],
    limit: int,
) -> list[tuple[Memory, float]]:
    ...
```

Use cosine distance:

```sql
embedding <=> :embedding
```

Return similarity as:

```python
similarity = 1.0 - distance
```

```python
async def fts_candidates(
    namespace: str,
    query: str,
    limit: int,
) -> list[tuple[Memory, float]]:
    ...
```

For Phase 1, use simple trigram similarity:

```sql
similarity(raw_content, :query)
```

Filter same namespace and `status = 'active'`.

```python
async def mark_accessed(memory_ids: list[UUID]) -> None:
    ...
```

Increment `access_count`, update `last_accessed_at`.

---

## 14. Graph store

Create `src/camillo/stores/graph_store.py`.

Implement:

```python
class GraphStore:
    def __init__(self, db: AsyncSession):
        self.db = db
```

Methods:

```python
async def create_or_increment_edge(
    source_id: UUID,
    target_id: UUID,
    increment: float = 1.0,
) -> None:
    ...
```

Rules:

* Do nothing if `source_id == target_id`.
* Store edges in canonical UUID order so `A-B` and `B-A` do not duplicate.
* If edge exists, increment weight and update `last_co_accessed_at`.
* If edge does not exist, insert it.

```python
async def reinforce_clique(memory_ids: list[UUID], increment: float = 1.0) -> None:
    ...
```

For every pair, call `create_or_increment_edge`.

No neighbor expansion yet in Phase 1.

---

## 15. Ingestion service

Create `src/camillo/cognitive/ingestion_service.py`.

Implement:

```python
class IngestionService:
    def __init__(
        self,
        memory_store: MemoryStore,
        graph_store: GraphStore,
        llm_service: LiteLLMService,
    ):
        ...
```

Method:

```python
async def ingest_interaction(
    namespace: str,
    user_msg: str,
    ai_msg: str,
    session_id: str | None,
) -> Memory:
    ...
```

Logic:

1. Build raw content:

```text
User:
{user_msg}

Assistant:
{ai_msg}
```

2. Score valence with LiteLLM.
3. Get embedding with LiteLLM.
4. Before insert, fetch previous memory in same namespace/session if session_id is not null.
5. Insert episodic memory.
6. If previous memory exists, create/increment Hebbian edge between previous and new memory.
7. Commit transaction in the API route, not inside the service unless necessary.

---

## 16. Recall service

Create `src/camillo/cognitive/recall_service.py`.

Implement a minimal Phase 1 recall.

```python
class RecallService:
    def __init__(
        self,
        memory_store: MemoryStore,
        graph_store: GraphStore,
        llm_service: LiteLLMService,
    ):
        ...
```

Method:

```python
async def recall(
    namespace: str,
    query: str,
    top_k: int,
) -> list[dict]:
    ...
```

Logic:

1. Embed query.
2. Get vector candidates, limit from settings.
3. Get FTS/trigram candidates, limit from settings.
4. Merge candidates by memory ID.
5. For each candidate, calculate:

```python
activation = calculate_activation(...)
score = 0.7 * retrieval_score + 0.3 * activation
```

6. Sort descending.
7. Take top K.
8. Mark returned memories as accessed.
9. Reinforce clique among returned memory IDs.
10. Return data for API response.

No LiteLLM rerank in Phase 1. No Hebbian spreading yet.

---

## 17. API routes

### `main.py`

Create FastAPI app:

```python
app = FastAPI(title=settings.app_name)
```

Include routers:

```python
health
ingest
recall
```

### `routes_health.py`

```python
GET /health
```

Return:

```json
{"status": "ok"}
```

### `routes_ingest.py`

```python
POST /ingest
```

Use DB dependency.

Create `MemoryStore`, `GraphStore`, `LiteLLMService`, `IngestionService`.

Call service.

Commit DB session.

Return `IngestResponse`.

### `routes_recall.py`

```python
POST /recall
```

Use DB dependency.

Create services.

Call recall.

Commit after reinforcement.

Return `RecallResponse`.

---

## 18. Tests

Create basic tests.

### `test_cognitive_math.py`

Test:

```python
activation is higher for recent memory than stale memory
activation increases with access_count
edge decay lowers old edge weight
activation is clamped
```

### `test_health.py`

Use FastAPI TestClient to verify:

```python
GET /health returns 200
```

Do not require Postgres for this test.

---

## 19. README

Create a concise README with:

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health
```

Example ingest:

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

Example recall:

```bash
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "repo:backend",
    "query": "What database did we choose?",
    "top_k": 5
  }'
```

---

## 20. Acceptance criteria

The project is complete when:

```bash
docker compose up --build
```

starts:

```text
postgres
db_migrator
cognitive_engine
```

and:

```bash
curl http://localhost:8000/health
```

returns:

```json
{"status":"ok"}
```

The code must be clean, typed where practical, and organized so later phases can add MCP, watcher, dreaming, full reranking, Hebbian spreading, and lifecycle management without rewriting Phase 1.

---

# Important Phase 1 Design Notes

For Phase 1, keep the system simple but structurally correct.

Do **not** implement:

```text
MCP
watcher
dreaming worker
full LiteLLM rerank
graph neighbor expansion
security redaction
soft delete lifecycle
memory events
```

But leave clear extension seams for all of them.

The most important Phase 1 outcome is not feature completeness. It is this:

```text
raw episodic memory can be ingested
embeddings are stored in pgvector
basic recall works
session-adjacent Hebbian edges are created
access reinforcement works
the system runs cleanly in Docker
```

---

# Phase 2 Preview: Recall Pipeline Expansion

Once Phase 1 is stable, Phase 2 should replace the basic recall service with the full cognitive recall pipeline:

```text
1. Embed query
2. Vector candidates
3. FTS/trigram candidates
4. RRF merge into top 30
5. LiteLLM rerank
6. Drop below relevance threshold
7. ACT-R activation filter
8. Final score = 0.7 relevance + 0.3 activation
9. Diversity pass
10. Top K
11. Hebbian spreading from top K
12. Async reinforcement
```

Add config:

```bash
RERANK_MIN_SCORE=0.35
HEBBIAN_SPREAD_LIMIT=3
HEBBIAN_MAX_DEPTH=1
DIVERSITY_SIMILARITY_THRESHOLD=0.92
```

---

# Phase 3 Preview: MCP

MCP should wrap your internal services rather than becoming a separate logic layer.

Tools:

```text
recall_context(query, namespace)
force_remember(text, namespace, type)
remember_interaction(user_msg, ai_msg, namespace, session_id)
forget_memory(memory_id)
memory_stats(namespace)
```

The MCP server should be thin:

```text
MCP request
→ service call
→ formatted result
```

---

# Phase 4 Preview: Dreaming Worker

Dreaming should operate on graph-connected raw episodic clusters, not random semantic similarity clusters.

Worker loop:

```text
every N minutes:
  select top activated episodic memories
  traverse Hebbian graph
  build clusters
  synthesize semantic rule with LiteLLM
  embed semantic rule
  store semantic memory
  lower source episodic base_importance
  write memory event records
```

Dream prompt:

```text
You are consolidating a cluster of raw, connected AI/user interactions.

Extract durable facts, preferences, project decisions, or recurring patterns.

Return a compact semantic memory rule.

Do not include temporary chatter.
Do not invent facts.
Preserve uncertainty when needed.
```

---

# My recommended next milestone

Start with the Codex prompt above exactly as Phase 1. After it builds, test these three flows:

```text
1. ingest one memory
2. ingest second memory in same session
3. recall by query
```

Then inspect Postgres:

```sql
select id, namespace, session_id, raw_content, base_importance, access_count
from memories;

select source_id, target_id, weight
from hebbian_edges;
```

That will confirm the cognitive substrate is alive before adding more intelligence.

[1]: https://hub.docker.com/r/pgvector/pgvector/tags?utm_source=chatgpt.com "pgvector - Docker Image"

