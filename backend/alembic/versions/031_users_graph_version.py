"""Add users.graph_version for OCC (P3 修 bug：图谱并发覆盖).

graph_memory.save_graphs 整列 rewrite users.knowledge_graph/error_graph 无版本保护，
两个并发 consolidate job（同用户不同课程，per-episode claim 不互斥同用户）会互相覆盖
（丢更新）。加 graph_version + 条件 UPDATE + 冲突重试（OCC），根治丢更新（宪法原则 5）。

不删图谱列--前端 GraphPage / 仪表盘 / 学生统计在用（删列=拆功能，已与用户确认保留）。
跨课串数据另修（教师 per-course 学生统计改读 knowledge_mastery）。

Revision ID: 030
Revises: 029
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "031"
down_revision: Union[str, None] = "030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _column_exists("users", "graph_version"):
        op.add_column(
            "users",
            sa.Column("graph_version", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    if _column_exists("users", "graph_version"):
        op.drop_column("users", "graph_version")
