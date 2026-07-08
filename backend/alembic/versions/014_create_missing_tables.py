"""Create core tables missing from the migration chain (C-2).

7 张核心表此前只由 init_db() 的 create_all 建立，迁移链从未 create_table：
knowledge_bases / kb_files / notebook_categories / notebook_entries /
notebook_entry_categories / bot_notifications / user_mcp_enrollments。

生产 init_db 跳过 create_all（database.py: ENVIRONMENT=production → return），纯靠
``alembic upgrade head`` → greenfield 部署缺这 7 张表，知识库 / 题目本 / Bot 通知 /
MCP 启用等模块全线 500（审计 C-2）。

本迁移幂等补建（_table_exists 守卫，对齐 001-013 既有风格）：
  - 已用 create_all 建过这些表的库（development / staging）跑此迁移为 no-op；
  - greenfield 生产新库则按 Base.metadata schema 建表。
列定义严格复刻 core/db/database.py 的模型（不带 server_default，对齐 create_all
产物，使两种部署方式产出的 schema 完全一致）。

Revision ID: 014
Revises: 013
Create Date: 2026-07-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    # -- knowledge_bases（owner_id 已含：002 仅在表已存在时补加该列，greenfield 表不存在跳过，
    #    故此处建表须带 owner_id；与 KnowledgeBase 模型一致）--
    if not _table_exists("knowledge_bases"):
        op.create_table(
            "knowledge_bases",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("course_id", sa.String(64), nullable=False),
            sa.Column("name", sa.String(256), nullable=False),
            sa.Column("description", sa.Text, nullable=False),
            sa.Column("icon", sa.String(32), nullable=False),
            sa.Column("system_prompt", sa.Text, nullable=False),
            sa.Column("sort_order", sa.Integer, nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("file_count", sa.Integer, nullable=False),
            sa.Column("error_msg", sa.Text, nullable=False),
            sa.Column("progress", sa.Integer, nullable=False),
            sa.Column("progress_msg", sa.Text, nullable=False),
            sa.Column("chunks_done", sa.Integer, nullable=False),
            sa.Column("chunks_total", sa.Integer, nullable=False),
            sa.Column("token_estimate", sa.Integer, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
            sa.Column("updated_at", sa.Float, nullable=False),
            sa.Column("is_visible", sa.Boolean, nullable=False),
            sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("join_code", sa.String(16), nullable=True),
        )
        # 模型 course_id / join_code 均为 unique=True, index=True → create_all 生成唯一索引
        op.create_index("ix_knowledge_bases_course_id", "knowledge_bases", ["course_id"], unique=True)
        op.create_index("ix_knowledge_bases_join_code", "knowledge_bases", ["join_code"], unique=True)

    # -- kb_files --
    if not _table_exists("kb_files"):
        op.create_table(
            "kb_files",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "kb_id",
                sa.String(32),
                sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("original_name", sa.String(512), nullable=False),
            sa.Column("file_path", sa.Text, nullable=False),
            sa.Column("file_size", sa.Integer, nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("error_msg", sa.Text, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_index("idx_kb_files_kb", "kb_files", ["kb_id", "created_at"])

    # -- notebook_categories --
    if not _table_exists("notebook_categories"):
        op.create_table(
            "notebook_categories",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
            # 表级 UNIQUE（对齐 create_all：模型 __table_args__ 的 UniqueConstraint）。
            # 不走 op.create_unique_constraint：SQLite 不支持 ALTER ADD CONSTRAINT，生产 PG 两者皆可。
            sa.UniqueConstraint("user_id", "name", name="uq_notebook_category_user_name"),
        )
        op.create_index("ix_notebook_categories_user_id", "notebook_categories", ["user_id"])

    # -- notebook_entries --
    if not _table_exists("notebook_entries"):
        op.create_table(
            "notebook_entries",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("session_id", sa.String(32), nullable=False),
            sa.Column("session_title", sa.String(256), nullable=False),
            sa.Column("question_id", sa.String(64), nullable=False),
            sa.Column("question", sa.Text, nullable=False),
            sa.Column("question_type", sa.String(32), nullable=False),
            sa.Column("options", sa.JSON, nullable=True),
            sa.Column("correct_answer", sa.Text, nullable=False),
            sa.Column("explanation", sa.Text, nullable=False),
            sa.Column("difficulty", sa.String(32), nullable=False),
            sa.Column("user_answer", sa.Text, nullable=False),
            sa.Column("is_correct", sa.Boolean, nullable=False),
            sa.Column("bookmarked", sa.Boolean, nullable=False),
            sa.Column("followup_session_id", sa.String(64), nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
            sa.Column("updated_at", sa.Float, nullable=False),
            sa.UniqueConstraint(
                "user_id", "session_id", "question_id", name="uq_notebook_entry_natural"
            ),
        )
        op.create_index("ix_notebook_entries_user_id", "notebook_entries", ["user_id"])
        op.create_index("idx_notebook_entries_user", "notebook_entries", ["user_id", "updated_at"])

    # -- notebook_entry_categories（M2M 联结，复合主键）--
    if not _table_exists("notebook_entry_categories"):
        op.create_table(
            "notebook_entry_categories",
            sa.Column(
                "entry_id",
                sa.Integer,
                sa.ForeignKey("notebook_entries.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "category_id",
                sa.Integer,
                sa.ForeignKey("notebook_categories.id", ondelete="CASCADE"),
                primary_key=True,
            ),
        )

    # -- bot_notifications --
    if not _table_exists("bot_notifications"):
        op.create_table(
            "bot_notifications",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("bot_id", sa.String(64), nullable=False),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("read", sa.Boolean, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_index("idx_bot_notif_user", "bot_notifications", ["user_id", "read", "created_at"])

    # -- user_mcp_enrollments --
    if not _table_exists("user_mcp_enrollments"):
        op.create_table(
            "user_mcp_enrollments",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("server_name", sa.String(64), nullable=False),
            sa.Column("enabled", sa.Boolean, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
            sa.UniqueConstraint("user_id", "server_name", name="uq_user_mcp_enrollment"),
        )
        op.create_index("idx_user_mcp_enrollment_user", "user_mcp_enrollments", ["user_id"])


def downgrade() -> None:
    # 反序 drop（按 FK 依赖：被引用表后删）。仅删本迁移新建的表。
    for table in (
        "user_mcp_enrollments",
        "bot_notifications",
        "notebook_entry_categories",
        "notebook_entries",
        "notebook_categories",
        "kb_files",
        "knowledge_bases",
    ):
        if _table_exists(table):
            op.drop_table(table)
