"""Add kb_builds table (per-backend index build state).

一门课程可同时构建 LightRAG 与 pgvector 两套索引：knowledge_bases 保持 1 行/课程
（继续挂 kb_files，文件两后端共享），新建 kb_builds 子表——每个 (kb_id, backend)
一行、status/progress 独立。UNIQUE(kb_id, backend) 保证每后端至多一条。

存量回填：每个 knowledge_bases 行按其 index_backend 生成一条 kb_builds，拷贝现有
status/progress/chunks_*/token_estimate/error_msg——老课程零变化（仍只一个后端），
新课程可分别构建两后端。knowledge_bases 上的旧 status/progress 列保留不删（避免破坏
性迁移），改由 kb_builds 派生（见 _kb_to_dict / _kb_to_course 的聚合），后续可单独清理。

Revision ID: 018
Revises: 017
Create Date: 2026-08-04

"""
import time
import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    # C-2 风格守卫：greenfield 新库 knowledge_bases 可能尚未建，此时建空 kb_builds 无意义，跳过。
    if not _table_exists("knowledge_bases"):
        print("[018] knowledge_bases absent (greenfield) — skip")
        return
    if not _table_exists("kb_builds"):
        op.create_table(
            "kb_builds",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "kb_id",
                sa.String(32),
                sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("backend", sa.String(32), nullable=False),  # lightrag | llamaindex_pg
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
            sa.Column("progress_msg", sa.Text, nullable=False, server_default=""),
            sa.Column("chunks_done", sa.Integer, nullable=False, server_default="0"),
            sa.Column("chunks_total", sa.Integer, nullable=False, server_default="0"),
            sa.Column("token_estimate", sa.Integer, nullable=False, server_default="0"),
            sa.Column("error_msg", sa.Text, nullable=False, server_default=""),
            sa.Column("updated_at", sa.Float, nullable=False),
            sa.UniqueConstraint("kb_id", "backend", name="uq_kb_builds_kb_backend"),
        )
        print("[018] created kb_builds")

    # 存量回填：仅当 kb_builds 为空时（首次升级），按每个 KB 的 index_backend 拷一条。
    bind = op.get_bind()
    existing = bind.execute(sa.text("SELECT count(*) FROM kb_builds")).scalar() or 0
    if existing == 0:
        rows = bind.execute(
            sa.text(
                "SELECT id, index_backend, status, progress, progress_msg, "
                "chunks_done, chunks_total, token_estimate, error_msg FROM knowledge_bases"
            )
        ).fetchall()
        now = time.time()
        builds = [
            {
                "id": uuid.uuid4().hex[:12],
                "kb_id": r[0],
                "backend": r[1] or "lightrag",
                "status": r[2] or "pending",
                "progress": r[3] or 0,
                "progress_msg": r[4] or "",
                "chunks_done": r[5] or 0,
                "chunks_total": r[6] or 0,
                "token_estimate": r[7] or 0,
                "error_msg": r[8] or "",
                "updated_at": now,
            }
            for r in rows
        ]
        if builds:
            builds_table = sa.table(
                "kb_builds",
                sa.Column("id", sa.String(32)),
                sa.Column("kb_id", sa.String(32)),
                sa.Column("backend", sa.String(32)),
                sa.Column("status", sa.String(32)),
                sa.Column("progress", sa.Integer),
                sa.Column("progress_msg", sa.Text),
                sa.Column("chunks_done", sa.Integer),
                sa.Column("chunks_total", sa.Integer),
                sa.Column("token_estimate", sa.Integer),
                sa.Column("error_msg", sa.Text),
                sa.Column("updated_at", sa.Float),
            )
            op.bulk_insert(builds_table, builds)
            print(f"[018] backfilled {len(builds)} kb_builds rows from knowledge_bases")


def downgrade() -> None:
    if _table_exists("kb_builds"):
        op.drop_table("kb_builds")
