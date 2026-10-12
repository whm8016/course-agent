"""补建 dev create_all 长出的 4 个孤儿列（全新生产库部署 P0）.

users.knowledge_graph / users.error_graph / users.is_admin 与 sessions.mode
从未进过迁移链：dev 流程靠 init_db() 的 create_all + _ensure_column 兜底建列，
production 显式跳过（database.py init_db：'Production relies on alembic'），
而迁移链里无人建这 4 列 → 全新库 alembic upgrade head 后首个写请求即
UndefinedColumnError（2026-10-12 阿里云生产首部署，注册接口 500 实测）。

与 032（course_topic 补票）、039（_column_exists 守卫）同款口径：
存在即跳过，幂等可重放。类型/默认值逐列对齐 init_db._ensure_column 的 DDL。

Revision ID: 040
Revises: 039
Create Date: 2026-10-12
"""

# revision identifiers, used by Alembic.
revision = "040"
down_revision = "039"
branch_labels: None = None
depends_on: None = None

from alembic import op
import sqlalchemy as sa


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _column_exists("users", "knowledge_graph"):
        op.add_column(
            "users",
            sa.Column(
                "knowledge_graph",
                sa.JSON(),
                nullable=False,
                server_default='{"nodes": [], "edges": []}',
            ),
        )
    if not _column_exists("users", "error_graph"):
        op.add_column(
            "users",
            sa.Column(
                "error_graph",
                sa.JSON(),
                nullable=False,
                server_default='{"nodes": [], "edges": []}',
            ),
        )
    if not _column_exists("users", "is_admin"):
        op.add_column(
            "users",
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        )
    if not _column_exists("sessions", "mode"):
        op.add_column(
            "sessions",
            sa.Column("mode", sa.String(32), nullable=False, server_default="chat"),
        )


def downgrade() -> None:
    if _column_exists("sessions", "mode"):
        op.drop_column("sessions", "mode")
    if _column_exists("users", "is_admin"):
        op.drop_column("users", "is_admin")
    if _column_exists("users", "error_graph"):
        op.drop_column("users", "error_graph")
    if _column_exists("users", "knowledge_graph"):
        op.drop_column("users", "knowledge_graph")
