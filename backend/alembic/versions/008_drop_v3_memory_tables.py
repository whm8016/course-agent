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
    # M-44：本迁移是破坏性的（DROP v3 三表），数据无法恢复，故不可逆。
    # 旧实现 ``pass`` 会让 ``alembic downgrade`` 静默「成功」——版本号回退了但表没回来，
    # 运维误以为已回滚。现显式 raise，强制人工介入（如确需重建空表骨架，DBA 手写 DDL）。
    raise NotImplementedError(
        "008 是破坏性迁移（DROP v3 memory 三表），数据已永久丢失，不可自动回滚。"
        " v3 子系统已被 Mem0 取代，schema 不再保留于代码库，无法精确重建。"
        " 如需回退版本号，请 DBA 手工 CREATE TABLE 重建空骨架后再 stamp。"
    )
