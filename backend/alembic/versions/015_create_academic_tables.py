"""Create academic tables: course_schedules + grades.

新增两张学业结构化数据表，配合 3 个只读学业查询工具（query_timetable /
query_grades / query_mistakes）与教师录入 API：
  - course_schedules：课程课表（课程级，不带 user_id），query_timetable 工具
    JOIN Enrollment 限定「我选的课」。
  - grades：学生成绩（学生级，student_id 强绑身份），query_grades 工具强制
    WHERE student_id==注入的 user_id。

幂等补建（_table_exists 守卫，对齐 014 既有风格）：
  - 已用 create_all 建过这些表的库（development / staging）跑此迁移为 no-op；
  - greenfield 生产新库则按 Base.metadata schema 建表。
列定义严格复刻 core/db/database.py 的模型（不带 server_default，对齐 create_all
产物，使两种部署方式产出的 schema 完全一致）。

Revision ID: 015
Revises: 014
Create Date: 2026-07-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    # -- course_schedules（课程课表，课程级无 user_id）--
    if not _table_exists("course_schedules"):
        op.create_table(
            "course_schedules",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("course_id", sa.String(64), nullable=False),
            sa.Column("weekday", sa.Integer, nullable=False),
            sa.Column("start_time", sa.String(16), nullable=False),
            sa.Column("end_time", sa.String(16), nullable=False),
            sa.Column("location", sa.String(128), nullable=False),
            sa.Column("teacher_name", sa.String(64), nullable=False),
            sa.Column("weeks", sa.String(64), nullable=False),
            sa.Column("note", sa.Text, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        # 模型 course_id 为 index=True → create_all 生成单列索引
        op.create_index("ix_course_schedules_course_id", "course_schedules", ["course_id"])
        op.create_index(
            "idx_course_schedule_course_weekday", "course_schedules", ["course_id", "weekday"]
        )

    # -- grades（学生成绩，学生级）--
    if not _table_exists("grades"):
        op.create_table(
            "grades",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "student_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("course_id", sa.String(64), nullable=False),
            sa.Column("item_name", sa.String(128), nullable=False),
            sa.Column("score", sa.Float, nullable=False),
            sa.Column("full_score", sa.Float, nullable=False),
            sa.Column("graded_at", sa.Float, nullable=True),
            sa.Column("comment", sa.Text, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
            sa.UniqueConstraint(
                "student_id", "course_id", "item_name", name="uq_grade_student_course_item"
            ),
        )
        op.create_index("ix_grades_student_id", "grades", ["student_id"])
        op.create_index("idx_grade_student_course", "grades", ["student_id", "course_id"])


def downgrade() -> None:
    # 反序 drop（按 FK 依赖：grades 引用 users，course_schedules 无外键；互不依赖）。
    for table in ("grades", "course_schedules"):
        if _table_exists(table):
            op.drop_table(table)
