"""掌握度估计器重构：mastery_observation 观测日志表 + Elo 折叠新列 + severity 改名。

感知与记账分离（plan「掌握度估计器重构」）：LLM 只做闭集序数判定写 append-only
观测日志（mastery_observation，唯一事实源），knowledge_mastery 退化为可重放的
确定性 Elo 折叠物化视图。

- mastery_observation：append-only，换估计器（Elo→BKT）只重跑折叠不重跑 LLM。
  字段契约参照 pyBKT 的 tutor log（student/skill/correct/opportunity）。
- knowledge_mastery.theta / n_obs：学生侧潜变量与观测数（折叠器写回）。
- course_topic.difficulty_b / n_item：题目难度参数（全班共享）与判分观测数。
  仅 source=quiz 的观测可更新（对话判定不冒充班级证据，见 plan §3b）。
- course_topic.severity → safety_weight：改名，避免与 error_patterns.severity 撞义
  （前者是 Task Risk 因子，后者是错误严重度）。032 未进生产，直接在 032 内改名，
  本迁移只对已有 severity 列的库做 RENAME 兜底。

全部幂等（_table_exists / _column_exists 守卫），与 032 同风格。

Revision ID: 033
Revises: 032
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "033"
down_revision: Union[str, None] = "032"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _table_exists("mastery_observation"):
        op.create_table(
            "mastery_observation",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id", sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("course_id", sa.String(64), nullable=False, default=""),
            sa.Column("topic_id", sa.String(64), nullable=False),
            sa.Column("source", sa.String(16), nullable=False),  # quiz / llm_judge / rule
            sa.Column("outcome", sa.Float(), nullable=False),    # quiz 0/1；good=1/fair=.5/poor=0
            sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("evidence_quote", sa.Text(), nullable=False, default=""),
            sa.Column("episode_ids", sa.JSON(), nullable=True),
            sa.Column("question_id", sa.String(64), nullable=True),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Index(
                "idx_mastery_obs_u_c_t_time",
                "user_id", "course_id", "topic_id", "created_at",
            ),
        )

    if not _column_exists("knowledge_mastery", "theta"):
        op.add_column(
            "knowledge_mastery",
            sa.Column("theta", sa.Float(), nullable=False, server_default="0.0"),
        )
    if not _column_exists("knowledge_mastery", "n_obs"):
        op.add_column(
            "knowledge_mastery",
            sa.Column("n_obs", sa.Integer(), nullable=False, server_default="0"),
        )

    if not _column_exists("course_topic", "difficulty_b"):
        op.add_column(
            "course_topic",
            sa.Column("difficulty_b", sa.Float(), nullable=False, server_default="0.0"),
        )
    if not _column_exists("course_topic", "n_item"):
        op.add_column(
            "course_topic",
            sa.Column("n_item", sa.Integer(), nullable=False, server_default="0"),
        )
    if not _column_exists("course_topic", "safety_weight"):
        op.add_column(
            "course_topic",
            sa.Column("safety_weight", sa.Float(), nullable=False, server_default="1.0"),
        )
        # severity → safety_weight 数据搬迁（旧库兜底；severity 保留只读兼容一个周期）
        if _column_exists("course_topic", "severity"):
            op.execute("UPDATE course_topic SET safety_weight = severity")


def downgrade() -> None:
    if _column_exists("course_topic", "safety_weight"):
        op.drop_column("course_topic", "safety_weight")
    if _column_exists("course_topic", "n_item"):
        op.drop_column("course_topic", "n_item")
    if _column_exists("course_topic", "difficulty_b"):
        op.drop_column("course_topic", "difficulty_b")
    if _column_exists("knowledge_mastery", "n_obs"):
        op.drop_column("knowledge_mastery", "n_obs")
    if _column_exists("knowledge_mastery", "theta"):
        op.drop_column("knowledge_mastery", "theta")
    if _table_exists("mastery_observation"):
        op.drop_table("mastery_observation")
