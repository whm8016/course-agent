"""统一的短码生成与归一化（无歧义字符集 + 向后兼容）。

覆盖历史上两套同源的短码：
- 教师邀请码  ``teacher_invites.code``
- 课程入课码  ``knowledge_bases.join_code``

两者都是需要人工键入的短 token，故共用一套**无歧义字符表**（剔除 0/O/1/I/L
等在无衬线字体和手写下易混的字形）。新码用该安全字母表生成。

向后兼容的关键设计：**归一化是 lossless 的**——``normalize_code`` 不校验字母表，
只去分隔符/空白并转大写。这样历史遗留的 hex 码（确实含 0/1）仍能原样匹配库内
存储值，**零数据迁移即可兼容存量码**。安全字母表只作用于 ``generate_code``。

关注点分离：generate（安全字母表）/ normalize（兼容归一）/ format（仅展示）三者
独立，互不耦合。

``ensure_unique_join_code`` 是唯一带副作用的函数——它调用 ``generate_code`` 后
查 ``knowledge_bases`` 表去重，是 DB 持久层辅助。admin/teacher 共用此函数，避免
任一方向 import 形成循环依赖（teacher 已 import admin 的 ``_kb_to_dict``）。
"""
from __future__ import annotations

import secrets
import string

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import KnowledgeBase

# Crockford-like 子集，剔除 0/O/1/I/L → 30 个无歧义字形。
ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
DEFAULT_LENGTH = 8

# 归一化时要剔除的字符：所有空白 + 连字符。
_STRIP_CHARS = string.whitespace + "-"


def generate_code(length: int = DEFAULT_LENGTH) -> str:
    """返回长度为 ``length`` 的密码学随机码。

    用 ``secrets``（非 ``random``）保证不可猜测——课程码授予入课权限、邀请码
    授予教师角色。默认长度 8 对应 30**8 ≈ 6.5e11 取值空间，百万级码的生日碰撞
    概率约 7e-7，因此调用方的去重循环只是纵深防御，不是主防线。
    """
    if length < 4:
        raise ValueError("code length must be >= 4")
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def normalize_code(raw: str | None) -> str:
    """把用户键入的码归一化为查库用的规范形式：去全部空白与连字符、转大写。

    幂等且完全（total）：``None``/空串返回 ``""``，便于调用方按真值分支判断。
    **不校验字母表**——旧 hex 码（含 0/1）与部分输入都能无损归一，保证存量
    选课记录继续可用。
    """
    if not raw:
        return ""
    return raw.translate(str.maketrans("", "", _STRIP_CHARS)).upper()


def format_code(raw: str | None) -> str:
    """人类可读的展示形式：大写、按 4-4 分组（``XXXX-XXXX``）。

    纯展示层，数据库永不存储连字符。非默认长度（如历史短码）原样大写返回，
    不会对奇数输入做错误切分。
    """
    norm = normalize_code(raw)
    if len(norm) == DEFAULT_LENGTH:
        return f"{norm[:4]}-{norm[4:]}"
    return norm


async def ensure_unique_join_code(
    db: AsyncSession, exclude_course_id: str | None = None
) -> str:
    """生成一个库内唯一的课程码（无歧义字符表）。

    最多重试 5 次（生日碰撞概率极低，5 次已是过度防御）。
    ``exclude_course_id`` 用于 refresh 场景排除自身课程；create_course 场景传
    None——新课程此时 join_code 为 NULL，不会与自身冲突。
    """
    for _ in range(5):
        code = generate_code(8)
        stmt = select(KnowledgeBase.id).where(KnowledgeBase.join_code == code)
        if exclude_course_id is not None:
            stmt = stmt.where(KnowledgeBase.course_id != exclude_course_id)
        clash = await db.execute(stmt)
        if not clash.first():
            return code
    raise HTTPException(status_code=500, detail="课程码生成失败，请重试")
