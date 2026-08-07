"""Convert the legacy namespace/scope model to a single-user corpus."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_single_user_memory"
down_revision: str | Sequence[str] | None = "0005_memory_scope"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_TYPE_MAP = {
    "episodic": "episode",
    "semantic": "fact",
    "relationship": "fact",
    "profile": "fact",
    "core": "fact",
    "preference": "preference",
    "procedural": "procedure",
}


def upgrade() -> None:
    """Preserve legacy memory data while removing obsolete isolation systems."""
    op.add_column("memories", sa.Column("workspace", sa.String(255), nullable=True))
    op.execute(
        "UPDATE memories SET workspace = substring(namespace FROM 6) WHERE namespace LIKE 'repo:%'"
    )
    op.execute("UPDATE memories SET workspace = namespace WHERE workspace IS NULL")
    op.execute(
        "UPDATE memories SET type = CASE type "
        "WHEN 'episodic' THEN 'episode' "
        "WHEN 'semantic' THEN 'fact' "
        "WHEN 'relationship' THEN 'fact' "
        "WHEN 'profile' THEN 'fact' "
        "WHEN 'core' THEN 'fact' "
        "WHEN 'preference' THEN 'preference' "
        "WHEN 'procedural' THEN 'procedure' ELSE type END"
    )
    op.drop_index("ix_memories_namespace_scope", table_name="memories")
    op.drop_index("ix_memories_scope", table_name="memories")
    op.drop_constraint("ck_memories_scope", "memories", type_="check")
    op.drop_column("memories", "scope")
    op.drop_index("ix_memories_namespace", table_name="memories")
    op.drop_column("memories", "namespace")
    op.drop_table("memory_relations")
    op.drop_table("hebbian_edges")
    op.drop_index("ix_dream_runs_namespace", table_name="dream_runs")
    op.drop_column("dream_runs", "namespace")
    op.create_index("ix_memories_active_workspace", "memories", ["status", "workspace"])
    op.create_index("ix_memories_workspace", "memories", ["workspace"])


def downgrade() -> None:
    """Restore legacy columns and tables where data can be reconstructed."""
    op.add_column("memories", sa.Column("namespace", sa.String(255), nullable=True))
    op.execute("UPDATE memories SET namespace = COALESCE(workspace, 'default')")
    op.alter_column("memories", "namespace", nullable=False)
    op.add_column("memories", sa.Column("scope", sa.Text(), nullable=False, server_default="local"))
    op.add_column(
        "dream_runs", sa.Column("namespace", sa.Text(), nullable=False, server_default="default")
    )
    op.drop_index("ix_memories_workspace", table_name="memories")
    op.drop_index("ix_memories_active_workspace", table_name="memories")
    op.drop_column("memories", "workspace")
