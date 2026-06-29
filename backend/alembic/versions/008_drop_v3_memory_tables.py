"""Drop legacy v3 memory tables.

The three-table v3 subsystem (memory_docs / memory_snapshot_state /
memory_consolidator_meta) is replaced by Mem0 (pip install mem0ai), whose
pgvector provider manages its own `memories` table at runtime — no project
migration needed for the new schema.

Revision ID: 008
Revises: 006
"""
from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 顺序：先有外键依赖的（均引用 users.id，无互相依赖，任意序皆可）
    op.execute("DROP TABLE IF EXISTS memory_consolidator_meta CASCADE")
    op.execute("DROP TABLE IF EXISTS memory_snapshot_state CASCADE")
    op.execute("DROP TABLE IF EXISTS memory_docs CASCADE")


def downgrade() -> None:
    # 旧架构已弃用，不重建
    pass
