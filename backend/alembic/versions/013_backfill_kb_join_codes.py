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


def _table_exists(table: str) -> bool:
    """检查表是否存在（对齐 001-014 的 _table_exists 守卫风格，幂等迁移）。"""
    bind = op.get_bind()
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    # C-2 greenfield 守卫：本迁移 SELECT FROM knowledge_bases，但 greenfield 新库该表
    # 由 014 创建（014 尚未执行到此）→ 旧逻辑抛 'no such table'，整个 upgrade head 中断。
    # 表不存在时安全跳过：greenfield 链中 014 紧随其后建表并留 join_code NULL（建库时已
    # 生成码，存量 NULL 行本就只存在于老库）。与 014 的 _table_exists 守卫同构。
    if not _table_exists("knowledge_bases"):
        print("[013] knowledge_bases absent (greenfield) — skip backfill")
        return

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
