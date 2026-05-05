"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-30 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the base memory and Hebbian edge tables.

    Returns:
        None. The migration mutates schema state through Alembic operations.
    """
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # This initial schema fixes embeddings at 1024 dimensions. Changing EMBEDDING_DIM
    # requires a new migration because pgvector dimensions are part of the column type.
    op.create_table(
        "memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False, server_default="episodic"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("base_importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_table(
        "hebbian_edges",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("last_co_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["memories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_id", "target_id"),
        sa.UniqueConstraint("source_id", "target_id", name="uq_hebbian_edges_pair"),
    )
    op.create_index("ix_memories_namespace", "memories", ["namespace"])
    op.create_index("ix_memories_session_id", "memories", ["session_id"])
    op.execute(
        "CREATE INDEX ix_memories_raw_content_trgm ON memories USING gin (raw_content gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Remove the base memory and Hebbian edge tables.

    Returns:
        None. The migration rolls the schema back through Alembic operations.
    """
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_memories_raw_content_trgm")
    op.drop_index("ix_memories_session_id", table_name="memories")
    op.drop_index("ix_memories_namespace", table_name="memories")
    op.drop_table("hebbian_edges")
    op.drop_table("memories")
