"""Backfill join_code for knowledge bases missing one.

管理员建库（admin.create_kb）历史上漏生成 join_code → 前端显示"暂无课程码"、
学生无法凭码入课。修复后新建库自动生成码；本迁移为存量 NULL 库（含管理员库与
内置课程库）逐个补码。幂等：仅处理 ``join_code IS NULL`` 的行，可重复执行。

逐行去重的原因：``knowledge_bases.join_code`` 上有部分唯一索引
（``ix_kb_join_code WHERE join_code IS NOT NULL``），非空码必须全表唯一，故不能
批量同值填充，需逐行 generate + 查重 + UPDATE。

Revision ID: 013
Revises: 012
Create Date: 2026-07-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    from core.codes import generate_code

    rows = bind.execute(
        sa.text("SELECT id FROM knowledge_bases WHERE join_code IS NULL")
    ).fetchall()

    backfilled = 0
    for (kb_id,) in rows:
        # 部分唯一索引要求非空码全表唯一 → 逐个生成并查重；5 次兜底，碰撞概率极低。
        code = None
        for _ in range(5):
            candidate = generate_code(8)
            clash = bind.execute(
                sa.text("SELECT 1 FROM knowledge_bases WHERE join_code = :c LIMIT 1"),
                {"c": candidate},
            ).first()
            if clash is None:
                code = candidate
                break
        if code is None:
            continue  # 5 次都撞（天文概率），跳过该行而非阻塞整个迁移
        bind.execute(
            sa.text("UPDATE knowledge_bases SET join_code = :c WHERE id = :id"),
            {"c": code, "id": kb_id},
        )
        backfilled += 1

    print(f"[013] backfilled join_code for {backfilled} knowledge_bases row(s)")


def downgrade() -> None:
    # backfill 不可逆：无法还原未知的 NULL；清空会破坏已分发给学生/教师的码。
    pass
