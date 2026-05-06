"""memory scope

Revision ID: 0005_memory_scope
Revises: 0004_dreaming
Create Date: 2026-05-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_memory_scope"
down_revision: str | Sequence[str] | None = "0004_dreaming"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add local/shared/global recall scope to memories.

    Returns:
        None. The migration mutates schema state through Alembic operations.
    """
    op.add_column(
        "memories",
        sa.Column("scope", sa.Text(), nullable=False, server_default="local"),
    )
    op.create_check_constraint(
        "ck_memories_scope",
        "memories",
        "scope IN ('local', 'shared', 'global')",
    )
    op.create_index("ix_memories_scope", "memories", ["scope"])
    op.create_index("ix_memories_namespace_scope", "memories", ["namespace", "scope"])


def downgrade() -> None:
    """Remove recall scope from memories.

    Returns:
        None. The migration rolls the schema back through Alembic operations.
    """
    op.drop_index("ix_memories_namespace_scope", table_name="memories")
    op.drop_index("ix_memories_scope", table_name="memories")
    op.drop_constraint("ck_memories_scope", "memories", type_="check")
    op.drop_column("memories", "scope")
