"""Question notebook REST API (aligned with question_notebook router)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.db.database import get_db
from core.db import notebook_store as nb
from core.db.limiter import limiter

router = APIRouter(prefix="/question/notebook", tags=["question_notebook"])


# --- Models ---


class CategoryItem(BaseModel):
    id: int
    name: str
    created_at: float = 0
    entry_count: int = 0


class NotebookEntryItem(BaseModel):
    id: int
    session_id: str
    session_title: str = ""
    question_id: str = ""
    question: str
    question_type: str = ""
    options: dict[str, str] = {}
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = ""
    user_answer: str = ""
    is_correct: bool = False
    grade: str = ""            # A3：good|fair|poor|""（未判/本地判）
    grade_source: str = ""     # A3：sandbox|llm_judge|local|""
    feedback: str = ""         # A3：判分反馈/未过用例摘要
    score: float | None = None # A3：coding 通过率
    bookmarked: bool = False
    followup_session_id: str = ""
    created_at: float
    updated_at: float
    categories: list[CategoryItem] | None = None


class NotebookEntryListResponse(BaseModel):
    items: list[NotebookEntryItem]
    total: int


class EntryUpdateRequest(BaseModel):
    bookmarked: bool | None = None
    followup_session_id: str | None = None


class CategoryCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CategoryRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CategoryAddRequest(BaseModel):
    category_id: int


class UpsertEntryRequest(BaseModel):
    session_id: str
    question_id: str
    question: str
    question_type: str = ""
    options: dict[str, str] | None = None
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = ""
    user_answer: str = ""
    is_correct: bool = False
    grade: str = ""            # A3：good|fair|poor|""（判分端点产出后随 upsert 落库）
    grade_source: str = ""     # A3：sandbox|llm_judge|local|""——"" 时守卫拦观测
    feedback: str = ""
    score: float | None = None


class GradeEntryRequest(BaseModel):
    """判分端点请求：一题一次（前端提交时逐题调用，60s 超时兜并发排队）。"""

    session_id: str = ""
    question_id: str = ""
    question_type: str = ""
    question: str = ""
    correct_answer: str = ""
    user_answer: str = ""
    test_cases: list[str] = Field(default_factory=list)
    starter_code: str = ""
    language: str = ""


# --- Entries ---


@router.post("/entries/upsert")
async def upsert_single_entry(
    payload: UpsertEntryRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    try:
        await nb.upsert_notebook_entry(db, uid, payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    entry = await nb.find_notebook_entry(db, uid, payload.session_id, payload.question_id)
    if entry is None:
        raise HTTPException(status_code=500, detail="Upsert failed")
    # A3 防污染守卫（替代前端单点约定）：已知非本地可判题型（written/short_answer/
    # coding/fill_in_blank 等无 options）只有在带判分来源（sandbox/llm_judge）时才
    # 回流学情事件与 mastery 观测——未判分传 is_correct=False 会污染错题过滤与掌握度。
    # 空 question_type 视为本地可判（legacy 兼容：answered 学情信号不能悄悄依赖
    # 客户端必传题型字段）。entry 照写（题面/作答/参考答案留存）。
    _NON_LOCAL_TYPES = ("written", "short_answer", "coding", "fill_in_blank")
    locally_gradable = not (
        payload.question_type in _NON_LOCAL_TYPES and not payload.options
    )
    graded_upstream = payload.grade_source in ("sandbox", "llm_judge")
    if payload.user_answer.strip() and (locally_gradable or graded_upstream):
        level = payload.grade if payload.grade in ("good", "fair", "poor") else (
            "good" if payload.is_correct else "poor"
        )
        # 观测权重：LLM 判分是软证据（W_JUDGE=0.6）；沙箱确定性结果与本地判同权（1.0）
        from core.memory.mastery_state import W_JUDGE, W_QUIZ

        weight = W_JUDGE if payload.grade_source == "llm_judge" else W_QUIZ
        # P0-a：学生提交作答 → 落一条 verb=answered 学情事件（L0，best-effort），喂
        # course_daily_rollup.answers。复用请求 db 提交（避免 SQLite 单连接另开会话
        # 死锁）；course_id 取自刚 upsert 的 entry（P1 已写时落盘）。
        from core.analytics.events import record_learning_event
        await record_learning_event(
            user_id=uid,
            course_id=entry.get("course_id", ""),
            verb="answered",
            object_id=payload.question_id,
            object_text=payload.question[:1000],
            session_id=payload.session_id,
            metadata={
                "is_correct": payload.is_correct,
                "difficulty": payload.difficulty,
                "grade": level,
                "grade_source": payload.grade_source,
            },
            db=db,
        )
        # 阶段F+A3：判分观测入 mastery_observation（题面经 match_topics 映射主题，
        # source=quiz——全班难度估计权重最高的来源，b 由 cron 日频重折）。best-effort，
        # 失败不阻塞答题主流程。level/weight/feedback：等级化回流（fair=0.5 部分正确）。
        try:
            from core.memory.quiz_observe import record_quiz_observation
            await record_quiz_observation(
                db, uid, entry.get("course_id", ""), payload.question_id,
                payload.question, level,
                user_answer=payload.user_answer,
                correct_answer=payload.correct_answer,
                weight=weight,
                feedback=payload.feedback,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "[notebook] quiz observation failed q=%s", payload.question_id, exc_info=True
            )
    return entry


@router.post("/grade")
@limiter.limit("10/minute")
async def grade_entry(
    request: Request,
    payload: GradeEntryRequest,
    user: dict = Depends(get_current_user),
):
    """判一题：coding→沙箱 run_tests（确定性），简答→LLM judge_answer。

    判分与写库解耦（判分失败不挡收录，可独立重试）；结果由前端随后随
    entries/upsert 落库并触发观测回流（守卫看 grade_source）。判不出用
    grade="ungraded" 表达，不用 5xx。限流 10/min 防 LLM 判分滥用。
    """
    from core.question.grading import grade_answer

    result = await grade_answer(
        payload.model_dump(), track={"user_id": str(user["id"])}
    )
    return {
        "question_id": result.question_id,
        "grade": result.grade,
        "score": result.score,
        "source": result.source,
        "feedback": result.feedback,
        "is_correct": result.is_correct,
        "test_results": result.test_results,
    }


@router.get("/entries", response_model=NotebookEntryListResponse)
async def list_entries(
    category_id: int | None = Query(default=None),
    bookmarked: bool | None = Query(default=None),
    is_correct: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotebookEntryListResponse:
    uid = str(user["id"])
    result = await nb.list_notebook_entries(
        db,
        uid,
        category_id=category_id,
        bookmarked=bookmarked,
        is_correct=is_correct,
        limit=limit,
        offset=offset,
    )
    return NotebookEntryListResponse(
        items=[NotebookEntryItem(**item) for item in result["items"]],
        total=result["total"],
    )


@router.get("/entries/lookup/by-question")
async def lookup_entry(
    session_id: str = Query(...),
    question_id: str = Query(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    entry = await nb.find_notebook_entry(db, uid, session_id, question_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.get("/entries/{entry_id}", response_model=NotebookEntryItem)
async def get_entry(
    entry_id: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotebookEntryItem:
    uid = str(user["id"])
    entry = await nb.get_notebook_entry(db, uid, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return NotebookEntryItem(**entry)


@router.patch("/entries/{entry_id}")
async def update_entry(
    entry_id: int,
    payload: EntryUpdateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updated = await nb.update_notebook_entry(db, uid, entry_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"updated": True, "id": entry_id}


@router.delete("/entries/{entry_id}")
async def delete_entry(
    entry_id: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    deleted = await nb.delete_notebook_entry(db, uid, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": True, "id": entry_id}


@router.post("/entries/{entry_id}/categories")
async def add_entry_to_category(
    entry_id: int,
    payload: CategoryAddRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    entry = await nb.get_notebook_entry(db, uid, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    ok = await nb.add_entry_to_category(db, uid, entry_id, payload.category_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to add to category")
    return {"added": True, "entry_id": entry_id, "category_id": payload.category_id}


@router.delete("/entries/{entry_id}/categories/{category_id}")
async def remove_entry_from_category(
    entry_id: int,
    category_id: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    removed = await nb.remove_entry_from_category(db, uid, entry_id, category_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Link not found")
    return {"removed": True, "entry_id": entry_id, "category_id": category_id}


# --- Categories ---


@router.get("/categories", response_model=list[CategoryItem])
async def list_categories(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    return await nb.list_categories(db, uid)


@router.post("/categories", response_model=CategoryItem, status_code=201)
async def create_category(
    payload: CategoryCreateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    try:
        return await nb.create_category(db, uid, payload.name)
    except ValueError:
        raise HTTPException(status_code=409, detail="Category name already exists") from None


@router.patch("/categories/{category_id}")
async def rename_category(
    category_id: int,
    payload: CategoryRenameRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    updated = await nb.rename_category(db, uid, category_id, payload.name)
    if not updated:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"updated": True, "id": category_id, "name": payload.name}


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    uid = str(user["id"])
    deleted = await nb.delete_category(db, uid, category_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"deleted": True, "id": category_id}
