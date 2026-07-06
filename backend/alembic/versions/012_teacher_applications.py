"""Add teacher_applications table (apply-approve admission flow).

教师准入从"邀请码即时升级"扩展为"申请-审批"流（与邀请码并存）。
新表 teacher_applications + 部分唯一索引（每用户至多一条 pending，并发安全
且允许 rejected 后重新申请）。

测试环境 init_db() 走 create_all 自建；生产由本迁移建表。

Revision ID: 012
Revises: 011
Create Date: 2026-07-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(bind, table: str) -> bool:
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if _table_exists(bind, "teacher_applications"):
        return  # 幂等：已建则跳过（开发态 create_all 可能已建）

    op.create_table(
        "teacher_applications",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", sa.String(32), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.Float, nullable=True),
        sa.Column("review_note", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.Float, nullable=False),
    )
    # 复合索引：服务"该用户是否有 pending"与"admin 按状态列表"两类查询。
    op.create_index(
        "ix_teacher_app_user_status", "teacher_applications", ["user_id", "status"]
    )
    op.create_index(
        "ix_teacher_app_status_created", "teacher_applications", ["status", "created_at"]
    )

    # 部分唯一索引：同一用户至多一条 pending（并发双击兜底），
    # 但不阻碍 rejected 后重新申请——只锁 pending 状态。PG/SQLite 均支持。
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_teacher_app_pending_user "
        "ON teacher_applications (user_id) WHERE status = 'pending'"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "teacher_applications"):
        return
    # drop_table 自动级联删除表上的所有索引（含部分唯一索引）。
    op.drop_table("teacher_applications")
