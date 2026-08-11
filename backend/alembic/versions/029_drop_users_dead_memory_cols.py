"""Drop dead memory columns from users (P3 users 瘦身).

按《表设计宪法》原则 1/6：users 表上的 summary_memory/profile_memory/scope_memory/
preferences_memory 是 L2 记忆早期方案的残留--语义记忆已交给 mem0（memories 表）、L2 摘要
落在 sessions.summary。这 4 列从无写入（永远默认空）；scope/preferences 甚至无读者，
summary/profile 仅 auth.py 返回（前端 types 标可选且无组件使用）。本次 Contract 删除。

knowledge_graph/error_graph 暂留--教师仪表盘 / GraphPage 前端直接渲染其 JSON（节点数、
高风险、错题数），删列需前端协同改造（见宪法 P3 路线图），本次不碰；其并发 rewrite 的
OCC 问题随列保留，待前端迁移后一并根治。

Revision ID: 029
Revises: 028
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "029"
down_revision: Union[str, None] = "028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


# (列名, 类型, 默认值) -- downgrade 按此恢复
_DEAD_COLS = [
    ("summary_memory", sa.Text, ""),
    ("profile_memory", sa.Text, "{}"),
    ("scope_memory", sa.Text, ""),
    ("preferences_memory", sa.Text, ""),
]


def upgrade() -> None:
    for name, _typ, _default in _DEAD_COLS:
        if _column_exists("users", name):
            op.drop_column("users", name)


def downgrade() -> None:
    for name, typ, default in _DEAD_COLS:
        if not _column_exists("users", name):
            op.add_column(
                "users",
                sa.Column(name, typ, nullable=False, server_default=default),
            )
