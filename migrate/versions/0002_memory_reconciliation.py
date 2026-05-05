"""memory reconciliation

Revision ID: 0002_memory_reconciliation
Revises: 0001_initial_schema
Create Date: 2026-05-04 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_memory_reconciliation"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add reconciliation fields and relation tracking tables.

    Returns:
        None. The migration mutates schema state through Alembic operations.
    """
    op.add_column(
        "memories",
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.8"),
    )
    op.add_column("memories", sa.Column("source", sa.Text(), nullable=True))
    op.add_column(
        "memories",
        sa.Column("superseded_by", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column("memories", sa.Column("deprecated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("memories", sa.Column("status_reason", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_memories_superseded_by_memories",
        "memories",
        "memories",
        ["superseded_by"],
        ["id"],
    )

    op.create_table(
        "memory_relations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["memories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_id"], ["memories.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "source_id",
            "target_id",
            "relation_type",
            name="uq_memory_relations_source_target_type",
        ),
    )
    op.create_index("ix_memories_status", "memories", ["status"])
    op.create_index("ix_memories_type", "memories", ["type"])
    op.create_index("ix_memories_superseded_by", "memories", ["superseded_by"])
    op.create_index("ix_memory_relations_source_id", "memory_relations", ["source_id"])
    op.create_index("ix_memory_relations_target_id", "memory_relations", ["target_id"])
    op.create_index("ix_memory_relations_relation_type", "memory_relations", ["relation_type"])


def downgrade() -> None:
    """Remove reconciliation fields and relation tracking tables.

    Returns:
        None. The migration rolls the schema back through Alembic operations.
    """
    op.drop_index("ix_memory_relations_relation_type", table_name="memory_relations")
    op.drop_index("ix_memory_relations_target_id", table_name="memory_relations")
    op.drop_index("ix_memory_relations_source_id", table_name="memory_relations")
    op.drop_index("ix_memories_superseded_by", table_name="memories")
    op.drop_index("ix_memories_type", table_name="memories")
    op.drop_index("ix_memories_status", table_name="memories")
    op.drop_table("memory_relations")
    op.drop_constraint("fk_memories_superseded_by_memories", "memories", type_="foreignkey")
    op.drop_column("memories", "status_reason")
    op.drop_column("memories", "deprecated_at")
    op.drop_column("memories", "superseded_by")
    op.drop_column("memories", "source")
    op.drop_column("memories", "confidence")
