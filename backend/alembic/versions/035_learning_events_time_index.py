"""learning_events 补 created_at 单列索引——rollup cron 时间窗扫描免全表扫。

既有两个复合索引 (course_id, verb, created_at) / (actor_user_id, course_id, created_at)
前导列都不是 created_at，而 rollup 查询是「created_at 时间窗 + course_id != ''
（否定谓词走不了任何前缀）+ GROUP BY course_id」——每小时 cron 对 append-only
大表全表扫 ×7 天窗口。本迁移只加一个单列索引修正访问路径。

幂等（_index_exists 守卫），与 034 同风格。

Revision ID: 035
Revises: 034
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "035"
down_revision: Union[str, None] = "034"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_exists(table: str, index: str) -> bool:
    bind = op.get_bind()
    return any(i["name"] == index for i in sa.inspect(bind).get_indexes(table))


def upgrade() -> None:
    if not _index_exists("learning_events", "idx_events_time"):
        op.create_index(
            "idx_events_time",
            "learning_events",
            ["created_at"],
        )


def downgrade() -> None:
    if _index_exists("learning_events", "idx_events_time"):
        op.drop_index("idx_events_time", table_name="learning_events")
