"""主题图抽取重做（课文锚定边）：course_topic_edge 加 evidence 列。

抽取主循环从「语义候选 + 两两判定」改为「块级同抽主题+课文明示先修边」（对齐
LightRAG/KGGen：原文随行、闭集端点、抽完再去重，见 course_topic_builder 模块头）。
evidence 落讲义原句（子串校验过）供教师核对悬停；闭包推断边/旧边为 NULL。

全部幂等（_column_exists 守卫），与 033 同风格。

Revision ID: 034
Revises: 033
Create Date: 2026-08-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "034"
down_revision: Union[str, None] = "033"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _column_exists("course_topic_edge", "evidence"):
        # nullable 无默认：旧边/推断边/判定边一律 NULL，只有课文锚定边有值（017 惯例）
        op.add_column(
            "course_topic_edge",
            sa.Column("evidence", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("course_topic_edge", "evidence"):
        op.drop_column("course_topic_edge", "evidence")
