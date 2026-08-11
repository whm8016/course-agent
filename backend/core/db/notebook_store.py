"""Async CRUD for question notebook (entries + categories)."""
from __future__ import annotations

import time
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import NotebookCategory, NotebookEntry, NotebookEntryCategory, Session as SessionModel


def _entry_to_dict(
    row: NotebookEntry,
    *,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    opts = row.options if row.options is not None else {}
    return {
        "id": row.id,
        "course_id": row.course_id,
        "session_id": row.session_id,
        "session_title": row.session_title,
        "question_id": row.question_id,
        "question": row.question,
        "question_type": row.question_type,
        "options": opts if isinstance(opts, dict) else {},
        "correct_answer": row.correct_answer,
        "explanation": row.explanation,
        "difficulty": row.difficulty,
        "user_answer": row.user_answer,
        "is_correct": row.is_correct,
        "bookmarked": row.bookmarked,
        "followup_session_id": row.followup_session_id,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "categories": categories or [],
    }


async def upsert_notebook_entry(
    db: AsyncSession, user_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    session_id = str(payload.get("session_id", ""))
    question_id = str(payload.get("question_id", ""))
    if not session_id or not question_id:
        raise ValueError("session_id and question_id are required")

    q = select(NotebookEntry).where(
        NotebookEntry.user_id == user_id,
        NotebookEntry.session_id == session_id,
        NotebookEntry.question_id == question_id,
    )
    res = await db.execute(q)
    existing = res.scalar_one_or_none()
    now = time.time()
    # P1：写时落盘 course_id（反查 Session，省去读侧 JOIN；宪法原则 3）
    course_id = (await db.execute(
        select(SessionModel.course_id).where(SessionModel.id == session_id)
    )).scalar_one_or_none() or ""

    fields = {
        "course_id": course_id,
        "session_title": str(payload.get("session_title", "")),
        "question": str(payload.get("question", "")),
        "question_type": str(payload.get("question_type", "")),
        "options": payload.get("options"),
        "correct_answer": str(payload.get("correct_answer", "")),
        "explanation": str(payload.get("explanation", "")),
        "difficulty": str(payload.get("difficulty", "")),
        "user_answer": str(payload.get("user_answer", "")),
        "is_correct": bool(payload.get("is_correct", False)),
        "updated_at": now,
    }

    if existing:
        for k, v in fields.items():
            setattr(existing, k, v)
        await db.flush()
        await db.refresh(existing)
        row = existing
    else:
        row = NotebookEntry(
            user_id=user_id,
            session_id=session_id,
            question_id=question_id,
            created_at=now,
            **fields,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)

    return _entry_to_dict(row)


async def find_notebook_entry(
    db: AsyncSession, user_id: str, session_id: str, question_id: str
) -> dict[str, Any] | None:
    q = select(NotebookEntry).where(
        NotebookEntry.user_id == user_id,
        NotebookEntry.session_id == session_id,
        NotebookEntry.question_id == question_id,
    )
    res = await db.execute(q)
    row = res.scalar_one_or_none()
    return _entry_to_dict(row) if row else None


async def get_notebook_entry(db: AsyncSession, user_id: str, entry_id: int) -> dict[str, Any] | None:
    q = select(NotebookEntry).where(NotebookEntry.user_id == user_id, NotebookEntry.id == entry_id)
    res = await db.execute(q)
    row = res.scalar_one_or_none()
    return _entry_to_dict(row) if row else None


async def list_notebook_entries(
    db: AsyncSession,
    user_id: str,
    *,
    category_id: int | None = None,
    bookmarked: bool | None = None,
    is_correct: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    conditions = [NotebookEntry.user_id == user_id]
    if bookmarked is not None:
        conditions.append(NotebookEntry.bookmarked == bookmarked)
    if is_correct is not None:
        conditions.append(NotebookEntry.is_correct == is_correct)

    count_stmt = select(func.count(NotebookEntry.id))
    list_stmt = select(NotebookEntry)
    if category_id is not None:
        join_on = NotebookEntryCategory.entry_id == NotebookEntry.id
        count_stmt = count_stmt.select_from(NotebookEntry).join(NotebookEntryCategory, join_on).where(
            and_(*conditions, NotebookEntryCategory.category_id == category_id)
        )
        list_stmt = (
            list_stmt.join(NotebookEntryCategory, join_on)
            .where(and_(*conditions, NotebookEntryCategory.category_id == category_id))
            .order_by(NotebookEntry.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
    else:
        count_stmt = count_stmt.where(and_(*conditions))
        list_stmt = (
            list_stmt.where(and_(*conditions))
            .order_by(NotebookEntry.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )

    total_res = await db.execute(count_stmt)
    total = int(total_res.scalar_one() or 0)

    res = await db.execute(list_stmt)
    rows = res.scalars().unique().all()
    items = [_entry_to_dict(r) for r in rows]
    return {"items": items, "total": total}


async def update_notebook_entry(
    db: AsyncSession, user_id: str, entry_id: int, updates: dict[str, Any]
) -> bool:
    allowed = {"bookmarked", "followup_session_id"}
    data = {k: v for k, v in updates.items() if k in allowed}
    if not data:
        return False
    data["updated_at"] = time.time()
    q = (
        update(NotebookEntry)
        .where(NotebookEntry.id == entry_id, NotebookEntry.user_id == user_id)
        .values(**data)
    )
    r = await db.execute(q)
    return r.rowcount > 0


async def delete_notebook_entry(db: AsyncSession, user_id: str, entry_id: int) -> bool:
    q = delete(NotebookEntry).where(NotebookEntry.id == entry_id, NotebookEntry.user_id == user_id)
    r = await db.execute(q)
    return r.rowcount > 0


async def list_categories(db: AsyncSession, user_id: str) -> list[dict[str, Any]]:
    q = (
        select(
            NotebookCategory.id,
            NotebookCategory.name,
            NotebookCategory.created_at,
            func.count(NotebookEntryCategory.entry_id).label("entry_count"),
        )
        .outerjoin(
            NotebookEntryCategory,
            NotebookEntryCategory.category_id == NotebookCategory.id,
        )
        .where(NotebookCategory.user_id == user_id)
        .group_by(NotebookCategory.id, NotebookCategory.name, NotebookCategory.created_at)
        .order_by(NotebookCategory.name)
    )
    res = await db.execute(q)
    out: list[dict[str, Any]] = []
    for row in res.all():
        out.append(
            {
                "id": row.id,
                "name": row.name,
                "created_at": row.created_at,
                "entry_count": int(row.entry_count or 0),
            }
        )
    return out


async def create_category(db: AsyncSession, user_id: str, name: str) -> dict[str, Any]:
    row = NotebookCategory(user_id=user_id, name=name.strip(), created_at=time.time())
    db.add(row)
    try:
        await db.flush()
    except IntegrityError as e:
        await db.rollback()
        raise ValueError("duplicate category name") from e
    await db.refresh(row)
    return {"id": row.id, "name": row.name, "created_at": row.created_at, "entry_count": 0}


async def rename_category(db: AsyncSession, user_id: str, category_id: int, name: str) -> bool:
    q = (
        update(NotebookCategory)
        .where(NotebookCategory.id == category_id, NotebookCategory.user_id == user_id)
        .values(name=name.strip())
    )
    r = await db.execute(q)
    return r.rowcount > 0


async def delete_category(db: AsyncSession, user_id: str, category_id: int) -> bool:
    # H-5：先校验 category 归属，非 owner 直接 return False，绝不碰连接表。
    # 旧实现先无条件删 NotebookEntryCategory 再删 category，攻击者传他人 category_id
    # 会把受害者的"题目-分类"关联全部抹掉（仅 category 因 user_id 过滤没删），造成数据破坏。
    own = await db.execute(
        select(NotebookCategory.id).where(
            NotebookCategory.id == category_id,
            NotebookCategory.user_id == user_id,
        )
    )
    if own.scalar_one_or_none() is None:
        return False
    await db.execute(
        delete(NotebookEntryCategory).where(NotebookEntryCategory.category_id == category_id)
    )
    q = delete(NotebookCategory).where(
        NotebookCategory.id == category_id,
        NotebookCategory.user_id == user_id,
    )
    r = await db.execute(q)
    return r.rowcount > 0


async def add_entry_to_category(
    db: AsyncSession, user_id: str, entry_id: int, category_id: int
) -> bool:
    e = await db.execute(
        select(NotebookEntry.id).where(NotebookEntry.id == entry_id, NotebookEntry.user_id == user_id)
    )
    if e.scalar_one_or_none() is None:
        return False
    c = await db.execute(
        select(NotebookCategory.id).where(
            NotebookCategory.id == category_id,
            NotebookCategory.user_id == user_id,
        )
    )
    if c.scalar_one_or_none() is None:
        return False
    exists = await db.execute(
        select(NotebookEntryCategory.entry_id).where(
            NotebookEntryCategory.entry_id == entry_id,
            NotebookEntryCategory.category_id == category_id,
        )
    )
    if exists.scalar_one_or_none() is not None:
        return True
    db.add(NotebookEntryCategory(entry_id=entry_id, category_id=category_id))
    await db.flush()
    return True


async def remove_entry_from_category(
    db: AsyncSession, user_id: str, entry_id: int, category_id: int
) -> bool:
    own = await db.execute(
        select(NotebookEntry.id).where(NotebookEntry.id == entry_id, NotebookEntry.user_id == user_id)
    )
    if own.scalar_one_or_none() is None:
        return False
    q = delete(NotebookEntryCategory).where(
        NotebookEntryCategory.entry_id == entry_id,
        NotebookEntryCategory.category_id == category_id,
    )
    r = await db.execute(q)
    return r.rowcount > 0
