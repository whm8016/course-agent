"""Add messages.turn_id + unique index (session_id, turn_id, role).

服务端权威回合日志（L2 返修步骤 1）：turn_runtime 每回合在 finally 落 user/assistant
两行，幂等键 = (session_id, turn_id, role)。此前 messages 行只由前端流结束后
best-effort 写入，关标签页/出错/取消轮永久丢失；改造后框架自己落盘（对标
Letta 服务端持有历史 / Agents SDK session_persistence）。

- turn_id nullable=True：存量行留 NULL，唯一索引中 NULL 互不冲突（PG/SQLite 同语义），
  单步安全加列，无需回填。
- 唯一「索引」而非表级约束：SQLite ALTER 不支持 add constraint，索引两方通用。
  重复投递由 add_message 的 SELECT 先查短路，索引是 DB 层兜底。

Revision ID: 027
Revises: 026
Create Date: 2026-09-09
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def _index_exists(table: str, index: str) -> bool:
    bind = op.get_bind()
    return index in [i["name"] for i in sa.inspect(bind).get_indexes(table)]


def upgrade() -> None:
    if not _column_exists("messages", "turn_id"):
        op.add_column("messages", sa.Column("turn_id", sa.String(length=32), nullable=True))
    if not _index_exists("messages", "uq_messages_session_turn_role"):
        op.create_index(
            "uq_messages_session_turn_role",
            "messages",
            ["session_id", "turn_id", "role"],
            unique=True,
        )


def downgrade() -> None:
    if _index_exists("messages", "uq_messages_session_turn_role"):
        op.drop_index("uq_messages_session_turn_role", table_name="messages")
    if _column_exists("messages", "turn_id"):
        op.drop_column("messages", "turn_id")
