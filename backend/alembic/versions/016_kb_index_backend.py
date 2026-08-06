"""Add knowledge_bases.index_backend column.

per-KB 索引后端选择：默认 'lightrag'（知识图谱，支持多跳关系推理），可选
'llamaindex_pg'（pgvector 快速向量检索，embedding 批调用分钟级建索引，与 LightRAG
二选一）。存量行回填 'lightrag'，行为零变化（新建 KB 默认仍是 lightrag）。

向量表（data_kb_chunks + HNSW + tsvector）不由本迁移建——交给 PGVectorStore 的
perform_setup 自动建（checkfirst=True 幂等，schema 跟着 llama-index-vector-stores-postgres
版本走，避免手写 migration 与其内部表结构耦合对不齐）。

Revision ID: 016
Revises: 015
Create Date: 2026-08-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    # C-2 风格守卫：greenfield 新库 knowledge_bases 可能尚未建（014 才补建），此时跳过。
    if not _table_exists("knowledge_bases"):
        print("[016] knowledge_bases absent (greenfield) — skip")
        return
    # server_default='lightrag' 回填存量行：所有已存在的 KB 默认走 LightRAG（不变）。
    if not _column_exists("knowledge_bases", "index_backend"):
        op.add_column(
            "knowledge_bases",
            sa.Column(
                "index_backend",
                sa.String(32),
                nullable=False,
                server_default="lightrag",
            ),
        )
        print("[016] added knowledge_bases.index_backend (default 'lightrag')")


def downgrade() -> None:
    if _table_exists("knowledge_bases") and _column_exists("knowledge_bases", "index_backend"):
        op.drop_column("knowledge_bases", "index_backend")
