"""Add kb_files.parser_engine column.

记录每个文件由哪个解析引擎解析（mineru_api / docling / mupdf），便于事后归因检索
质量问题（同一文件不同引擎产出差异）。nullable——解析层未启用或未解析时为空。
与 016（index_backend）独立，可单独回滚。

Revision ID: 017
Revises: 016
Create Date: 2026-08-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _table_exists("kb_files"):
        print("[017] kb_files absent (greenfield) — skip")
        return
    if not _column_exists("kb_files", "parser_engine"):
        op.add_column(
            "kb_files",
            sa.Column("parser_engine", sa.String(32), nullable=True),
        )
        print("[017] added kb_files.parser_engine")


def downgrade() -> None:
    if _table_exists("kb_files") and _column_exists("kb_files", "parser_engine"):
        op.drop_column("kb_files", "parser_engine")
