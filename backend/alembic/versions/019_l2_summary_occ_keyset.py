"""Add L2 summary OCC version + keyset cursor columns.

为 L2 会话摘要修三个架构缺陷（见 session_summary.py）：
- summary_version（Integer NOT NULL, server_default 0）：乐观并发控制版本号。
  写回走条件 UPDATE WHERE summary_version = :old，rowcount=0 判冲突、放弃本轮，
  保证多 worker 并发压缩不互相覆盖（取代原无条件 UPDATE 的 lost update）。
- summary_up_to_created_at（Float, nullable）：keyset 游标的时间分量，配合
  summary_up_to_msg_id（字符串 tie-break）把增量区间改成 SQL 范围查询
  WHERE (created_at, id) > (cursor_ts, cursor_id)，消除每轮全量拉消息。

存量行 summary_up_to_created_at 为 NULL，升级后首次压缩走一次 msg_id 兼容路径
（按现有 id 定位游标），成功后写入新游标，此后全部走 SQL 路径。summary_up_to_msg_id
保留不动，继续作 tie-break 与向后兼容。小表加列，SQLite（NOT NULL 需常量默认值，
server_default="0" 满足）与 PG 都安全。

Revision ID: 019
Revises: 018
Create Date: 2026-08-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # OCC 版本号：NOT NULL，存量行由 server_default 回填为 0
    op.add_column(
        "sessions",
        sa.Column("summary_version", sa.Integer(), nullable=False, server_default="0"),
    )
    # keyset 游标时间分量：存量行为 NULL（首次压缩走 msg_id 兼容路径后回填）
    op.add_column(
        "sessions",
        sa.Column("summary_up_to_created_at", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sessions", "summary_up_to_created_at")
    op.drop_column("sessions", "summary_version")
