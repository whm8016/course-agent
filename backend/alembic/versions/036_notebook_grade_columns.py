"""notebook_entries 补判分四列——grade/grade_source/feedback/score（A3 判分闭环）。

is_correct 布尔语义保留（fair→False=「未全对进错题本」），等级/来源/反馈/通过率
由新列承载：grade(good|fair|poor|"")、grade_source(sandbox|llm_judge|local|"")、
feedback（判分一句话反馈/未过用例摘要）、score（coding 通过率，可空）。

幂等（_col_exists 守卫），与 035 同风格。

Revision ID: 036
Revises: 035
Create Date: 2026-08-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "036"
down_revision: Union[str, None] = "035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLS = [
    sa.Column("grade", sa.String(16), nullable=False, server_default=""),
    sa.Column("grade_source", sa.String(16), nullable=False, server_default=""),
    sa.Column("feedback", sa.Text(), nullable=False, server_default=""),
    sa.Column("score", sa.Float(), nullable=True),
]


def _existing_cols(table: str) -> set[str]:
    bind = op.get_bind()
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    existing = _existing_cols("notebook_entries")
    for col in _COLS:
        if col.name not in existing:
            op.add_column("notebook_entries", col)


def downgrade() -> None:
    existing = _existing_cols("notebook_entries")
    for name in ("score", "feedback", "grade_source", "grade"):
        if name in existing:
            op.drop_column("notebook_entries", name)
