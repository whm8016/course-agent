"""Drop dead build-status columns from knowledge_bases (P2 KB 真相源收口).

按《表设计宪法》原则 1（一概念一真相源）：KB 构建状态的权威存储是 kb_builds（每后端一行，
018 引入）。knowledge_bases 行上的旧 status/progress/progress_msg/chunks_done/chunks_total/
token_estimate/error_msg 自 018 起不再被索引流程写入、也不被任何读路径消费--
_kb_to_dict 改读 aggregate_build_status(kb.builds) + _primary_build(...)。本次 Contract 删除。

保留 file_count（kb_builds 无此列，仍是 KB 行活字段，teacher.py 数文件数用）。

downgrade 恢复旧列（带 server_default 回填存量行，还原 018 前行为）。

Revision ID: 028
Revises: 027
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "028"
down_revision: Union[str, None] = "027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


# (列名, 类型, 018 前默认值)--downgrade 按此恢复
_DEAD_COLS = [
    ("status", sa.String(32), "pending"),
    ("progress", sa.Integer, 0),
    ("progress_msg", sa.Text, ""),
    ("chunks_done", sa.Integer, 0),
    ("chunks_total", sa.Integer, 0),
    ("token_estimate", sa.Integer, 0),
    ("error_msg", sa.Text, ""),
]


def upgrade() -> None:
    for name, _typ, _default in _DEAD_COLS:
        if _column_exists("knowledge_bases", name):
            op.drop_column("knowledge_bases", name)


def downgrade() -> None:
    for name, typ, default in _DEAD_COLS:
        if not _column_exists("knowledge_bases", name):
            op.add_column(
                "knowledge_bases",
                sa.Column(name, typ, nullable=False, server_default=str(default) if default != "" else ""),
            )
