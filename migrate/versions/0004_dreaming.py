"""dreaming consolidation

Revision ID: 0004_dreaming
Revises: 0002_memory_reconciliation
Create Date: 2026-05-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_dreaming"
down_revision: str | Sequence[str] | None = "0002_memory_reconciliation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create dreaming run audit storage and active episodic lookup indexes.

    Returns:
        None. The migration mutates schema state through Alembic operations.
    """
    op.create_table(
        "dream_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "seed_memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column(
            "source_memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column(
            "created_memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("clusters_considered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("clusters_dreamed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("memories_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("ix_dream_runs_namespace", "dream_runs", ["namespace"])
    op.create_index("ix_dream_runs_status", "dream_runs", ["status"])
    op.create_index("ix_dream_runs_started_at", "dream_runs", ["started_at"])
    op.execute("CREATE INDEX IF NOT EXISTS ix_memories_status ON memories(status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_memories_type_status ON memories(type, status)")


def downgrade() -> None:
    """Remove dreaming run audit storage.

    Returns:
        None. The migration rolls the schema back through Alembic operations.
    """
    op.execute("DROP INDEX IF EXISTS ix_memories_type_status")
    op.drop_index("ix_dream_runs_started_at", table_name="dream_runs")
    op.drop_index("ix_dream_runs_status", table_name="dream_runs")
    op.drop_index("ix_dream_runs_namespace", table_name="dream_runs")
    op.drop_table("dream_runs")
