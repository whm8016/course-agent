"""课程闭包零 LLM 记忆层：course_topic 两表生产迁移 + mastery/episodes/topic 新列。

一次加齐（避免二次迁移）：
- course_topic / course_topic_edge：模型早已有（create_all 自建），生产库此前无迁移——
  本迁移补上。severity 一并建进表（Task Risk 因子，默认 1.0 = 行为与不加列一致）。
- knowledge_mastery.strength：MemoryBank 遗忘强度 S（默认 5.0 天），门控 p_forget 用。
- knowledge_mastery.last_stitched_at：VoI cooldown 键（user+course+topic 上次实际拼出时间）。
- memory_episodes.topic_ids：巩固时 match_topics 命中的 topic_id 列表（段级回填）。

全部幂等（_table_exists / _column_exists 守卫），与 031 同风格。

Revision ID: 032
Revises: 031
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "032"
down_revision: Union[str, None] = "031"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _table_exists("course_topic"):
        op.create_table(
            "course_topic",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("course_id", sa.String(64), nullable=False, server_default=""),
            sa.Column("topic_id", sa.String(64), nullable=False),
            sa.Column("label", sa.String(128), nullable=False, server_default=""),
            sa.Column("definition", sa.Text(), nullable=False, server_default=""),
            sa.Column("source_section", sa.String(255), nullable=False, server_default=""),
            sa.Column("order_idx", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("embedding", sa.JSON(), nullable=True),
            sa.Column("severity", sa.Float(), nullable=False, server_default="1.0"),
            sa.UniqueConstraint("course_id", "topic_id", name="uq_course_topic"),
        )
        op.create_index("idx_course_topic_course", "course_topic", ["course_id"])
    elif not _column_exists("course_topic", "severity"):
        op.add_column(
            "course_topic",
            sa.Column("severity", sa.Float(), nullable=False, server_default="1.0"),
        )

    if not _table_exists("course_topic_edge"):
        op.create_table(
            "course_topic_edge",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("course_id", sa.String(64), nullable=False, server_default=""),
            sa.Column("src_topic_id", sa.String(64), nullable=False),
            sa.Column("dst_topic_id", sa.String(64), nullable=False),
            sa.Column("relation", sa.String(32), nullable=False, server_default="prerequisite"),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("verified_by", sa.String(64), nullable=True),
            sa.UniqueConstraint(
                "course_id", "src_topic_id", "dst_topic_id", "relation",
                name="uq_course_topic_edge",
            ),
        )
        op.create_index(
            "idx_course_topic_edge_dst", "course_topic_edge", ["course_id", "dst_topic_id"]
        )

    if not _column_exists("knowledge_mastery", "strength"):
        op.add_column(
            "knowledge_mastery",
            sa.Column("strength", sa.Float(), nullable=False, server_default="5.0"),
        )
    if not _column_exists("knowledge_mastery", "last_stitched_at"):
        op.add_column("knowledge_mastery", sa.Column("last_stitched_at", sa.Float(), nullable=True))
    if not _column_exists("memory_episodes", "topic_ids"):
        op.add_column("memory_episodes", sa.Column("topic_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    if _column_exists("memory_episodes", "topic_ids"):
        op.drop_column("memory_episodes", "topic_ids")
    if _column_exists("knowledge_mastery", "last_stitched_at"):
        op.drop_column("knowledge_mastery", "last_stitched_at")
    if _column_exists("knowledge_mastery", "strength"):
        op.drop_column("knowledge_mastery", "strength")
    if _table_exists("course_topic_edge"):
        op.drop_table("course_topic_edge")
    if _table_exists("course_topic"):
        op.drop_table("course_topic")
