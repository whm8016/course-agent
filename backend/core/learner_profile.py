"""Learner memory (DeepTutor 风格的两文件画像系统，DB 列承载).

本模块把原来的「关键词匹配 + 定字段 JSON 画像 + 最近 25 行问答日志」升级成
DeepTutor 的方案：两份自由 Markdown（画像 PROFILE + 学习摘要 SUMMARY），
每隔若干轮由 LLM 重写一次，不变则返回 NO_CHANGE。

存储：
  users.profile_memory  -> Markdown 文本（原为 JSON，首次读取时自动迁移）
  users.summary_memory  -> Markdown 文本

对外主要 API：
  build_memory_context(user)                                -> str
  update_learner_memory(db, user_id, *, course_id, mode,
                         user_message, assistant_answer)     -> None
  refresh_from_source(db, user_id, source, language="zh")    -> dict
  read_snapshot(db, user_id)                                 -> dict
  write_file(db, user_id, which, content)                    -> dict
  clear_memory(db, user_id, which=None)                      -> dict
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import TEXT_MODEL
from core.database import User
from core.llm import client as async_openai_client

logger = logging.getLogger(__name__)

MemoryFile = Literal["summary", "profile"]
MEMORY_FILES: tuple[MemoryFile, ...] = ("summary", "profile")

_NO_CHANGE = "NO_CHANGE"
_REWRITE_EVERY_N_TURNS = 3
_MAX_CONTEXT_CHARS = 4000

# 计数 key：用于判断何时触发 LLM 重写；保存在 profile_memory 的首行注释中
_COUNTER_KEY = "<!-- turn_counter:"


# ---------------------------------------------------------------------------
# 存储读写（两份字段都按 Markdown 处理；兼容旧 JSON 画像）
# ---------------------------------------------------------------------------

def _legacy_profile_to_markdown(raw: str) -> str:
    """把旧版 {level, style, goal, preferred_mode} JSON 画像转成 Markdown。

    当 raw 不是 JSON 或空时返回空串；上层会忽略空值。
    """
    if not raw:
        return ""
    stripped = raw.strip()
    if not stripped.startswith("{"):
        return stripped  # 已经是 Markdown
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(data, dict):
        return ""
    lines: list[str] = []
    identity = []
    level = str(data.get("level") or "").strip()
    style = str(data.get("style") or "").strip()
    goal = str(data.get("goal") or "").strip()
    mode = str(data.get("preferred_mode") or "").strip()
    if level and level != "unknown":
        identity.append(f"- 当前水平：{level}")
    if goal:
        identity.append(f"- 主要目标：{goal}")
    if identity:
        lines.append("## Identity")
        lines.extend(identity)
    prefs = []
    if style:
        prefs.append(f"- 讲解风格：{style}")
    if mode:
        prefs.append(f"- 偏好模式：{mode}")
    if prefs:
        lines.append("## Preferences")
        lines.extend(prefs)
    return "\n".join(lines).strip()


def _read_counter(profile_raw: str) -> tuple[int, str]:
    """从 profile Markdown 中解析 turn counter（首行 HTML 注释）。返回 (计数, 去注释后的正文)。"""
    if not profile_raw:
        return 0, ""
    text = profile_raw.lstrip()
    if text.startswith(_COUNTER_KEY):
        m = re.match(r"<!--\s*turn_counter:(\d+)\s*-->\s*\n?", text)
        if m:
            try:
                return int(m.group(1)), text[m.end():]
            except ValueError:
                return 0, text[m.end():]
    return 0, profile_raw


def _write_counter(body: str, counter: int) -> str:
    """把 counter 写回 profile Markdown 首行注释；body 为空时只保留计数标记。"""
    body_clean = (body or "").lstrip()
    if body_clean.startswith(_COUNTER_KEY):
        body_clean = re.sub(r"<!--\s*turn_counter:\d+\s*-->\s*\n?", "", body_clean, count=1)
    header = f"<!-- turn_counter:{max(0, counter)} -->\n"
    return header + body_clean.strip()


async def _load_fields(db: AsyncSession, user_id: str, *, lock: bool = False) -> tuple[str, str] | None:
    stmt = select(User.summary_memory, User.profile_memory).where(User.id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    row = (await db.execute(stmt)).first()
    if not row:
        return None
    return (row.summary_memory or "").strip(), (row.profile_memory or "").strip()


async def _save_fields(
    db: AsyncSession,
    user_id: str,
    *,
    summary: str | None = None,
    profile: str | None = None,
) -> None:
    values: dict = {}
    if summary is not None:
        values["summary_memory"] = summary.strip()
    if profile is not None:
        values["profile_memory"] = profile.strip()
    if not values:
        return
    await db.execute(update(User).where(User.id == user_id).values(**values))


# ---------------------------------------------------------------------------
# build_memory_context：把画像+摘要注入 system prompt
# ---------------------------------------------------------------------------

def build_memory_context(user: dict | None, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """根据 user dict（auth 返回的 summary_memory / profile_memory）生成注入文本。

    兼容三种情况：
    - profile_memory 是新版 Markdown 字符串
    - profile_memory 是旧版 dict（早期前端/后端解析结果）
    - profile_memory 是旧版 JSON 字符串
    """
    if not user:
        return ""

    summary = str(user.get("summary_memory") or "").strip()

    profile_raw = user.get("profile_memory") or ""
    if isinstance(profile_raw, dict):
        profile_raw = json.dumps(profile_raw, ensure_ascii=False)
    profile_md = _legacy_profile_to_markdown(str(profile_raw))
    _, profile_md = _read_counter(profile_md)
    profile_md = profile_md.strip()

    parts: list[str] = []
    if profile_md:
        parts.append(f"### 用户画像（User Profile）\n{profile_md}")
    if summary:
        parts.append(f"### 学习轨迹（Learning Context）\n{summary}")

    if not parts:
        return ""

    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n...[truncated]"

    return (
        "## 背景记忆（仅在直接相关时参考，不要硬塞进答案）\n"
        f"{combined}"
    )


# ---------------------------------------------------------------------------
# LLM 重写
# ---------------------------------------------------------------------------

def _profile_prompts(current: str, source: str) -> tuple[str, str]:
    system = (
        "你负责维护一份学生画像文档。只保留稳定的身份信息、学习风格、知识水平和偏好，"
        "不要记录一次性聊天内容或临时话题。"
        f"如果无需修改，请只输出 {_NO_CHANGE}，不要输出任何其他字符。"
    )
    user = (
        "请在需要时重写完整画像，使用如下 Markdown 小节（可按需增删）：\n"
        "## Identity\n## Learning Style\n## Knowledge Level\n## Preferences\n\n"
        "规则：整体保持简洁、去掉过时或矛盾的条目、用短句，不要写会话流水。\n\n"
        f"[当前画像]\n{current or '(empty)'}\n\n"
        f"[新增材料]\n{source}"
    )
    return system, user


def _summary_prompts(current: str, source: str) -> tuple[str, str]:
    system = (
        "你负责维护一份学习轨迹摘要，记录学生正在学什么、已经掌握什么、还遗留哪些问题。"
        f"如果无需修改，请只输出 {_NO_CHANGE}，不要输出任何其他字符。"
    )
    user = (
        "请在需要时重写完整摘要，使用如下 Markdown 小节（可按需增删）：\n"
        "## Current Focus\n## Accomplishments\n## Open Questions\n\n"
        "规则：整体保持简洁、删除已完成或过时的条目、用短句，不要写逐轮对白。\n\n"
        f"[当前摘要]\n{current or '(empty)'}\n\n"
        f"[新增材料]\n{source}"
    )
    return system, user


def _strip_code_fence(content: str) -> str:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


async def _rewrite_one(which: MemoryFile, current: str, source: str) -> tuple[bool, str]:
    """调用 LLM 重写单份文档。返回 (是否有变更, 最新正文)."""
    if which == "profile":
        sys_prompt, user_prompt = _profile_prompts(current, source)
    else:
        sys_prompt, user_prompt = _summary_prompts(current, source)

    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
            stream=False,
        )
    except Exception as e:
        logger.warning("memory rewrite LLM call failed which=%s err=%s", which, e)
        return False, current

    raw = ""
    try:
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw = ""
    raw = _strip_code_fence(raw)

    if not raw or raw == _NO_CHANGE:
        return False, current
    if raw == current.strip():
        return False, current
    return True, raw


# ---------------------------------------------------------------------------
# 对外：按轮触发的 memory 刷新
# ---------------------------------------------------------------------------

def _build_source_for_turn(
    *,
    course_id: str,
    mode: str,
    user_message: str,
    assistant_answer: str,
) -> str:
    return (
        f"[Course] {course_id or '(unknown)'}\n"
        f"[Capability] {mode or 'chat'}\n"
        f"[Timestamp] {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"[User]\n{(user_message or '').strip()}\n\n"
        f"[Assistant]\n{(assistant_answer or '').strip()}"
    )


async def update_learner_memory(
    db: AsyncSession,
    user_id: str,
    *,
    course_id: str,
    mode: str,
    user_message: str,
    assistant_answer: str,
    force: bool = False,
) -> None:
    """每轮对话结束后调用；每 N 轮真正跑一次 LLM 重写（force=True 立即刷新）。

    对齐 DeepTutor `MemoryService.refresh_from_turn`，但把"文件"换成 DB 列，
    并在 profile_memory 的首行 HTML 注释里维护 turn_counter。
    """
    if not user_message.strip() or not assistant_answer.strip():
        return

    loaded = await _load_fields(db, user_id, lock=True)
    if loaded is None:
        return
    summary_cur, profile_raw = loaded

    # 把旧 JSON 迁成 Markdown，再抽出 counter
    profile_md_with_counter = _legacy_profile_to_markdown(profile_raw)
    counter, profile_cur = _read_counter(profile_md_with_counter)
    counter += 1

    if not force and counter < _REWRITE_EVERY_N_TURNS:
        # 没到阈值：只更新计数，不打 LLM
        await _save_fields(db, user_id, profile=_write_counter(profile_cur, counter))
        return

    source = _build_source_for_turn(
        course_id=course_id,
        mode=mode,
        user_message=user_message,
        assistant_answer=assistant_answer,
    )

    p_changed, new_profile = await _rewrite_one("profile", profile_cur, source)
    s_changed, new_summary = await _rewrite_one("summary", summary_cur, source)

    # 重写成功后 counter 归零；失败也归零避免长期堆积（下次达阈值重试）
    final_profile = _write_counter(new_profile if p_changed else profile_cur, 0)
    final_summary = new_summary if s_changed else summary_cur
    await _save_fields(db, user_id, summary=final_summary, profile=final_profile)

    logger.info(
        "memory refreshed user=%s profile_changed=%s summary_changed=%s",
        user_id, p_changed, s_changed,
    )


# ---------------------------------------------------------------------------
# 对外：API 路由用到的读/写/清
# ---------------------------------------------------------------------------

async def read_snapshot(db: AsyncSession, user_id: str) -> dict:
    loaded = await _load_fields(db, user_id)
    if loaded is None:
        return {"summary": "", "profile": ""}
    summary_cur, profile_raw = loaded
    profile_md = _legacy_profile_to_markdown(profile_raw)
    _, profile_cur = _read_counter(profile_md)
    return {"summary": summary_cur, "profile": profile_cur.strip()}


async def write_file(
    db: AsyncSession, user_id: str, which: MemoryFile, content: str,
) -> dict:
    if which not in MEMORY_FILES:
        raise ValueError(f"invalid memory file: {which}")
    normalized = (content or "").strip()
    if which == "profile":
        # 保留 counter（重置为 0，避免人工编辑后马上被覆写）
        await _save_fields(db, user_id, profile=_write_counter(normalized, 0))
    else:
        await _save_fields(db, user_id, summary=normalized)
    return await read_snapshot(db, user_id)


async def clear_memory(
    db: AsyncSession, user_id: str, which: MemoryFile | None = None,
) -> dict:
    if which is None:
        await _save_fields(db, user_id, summary="", profile=_write_counter("", 0))
    elif which == "profile":
        await _save_fields(db, user_id, profile=_write_counter("", 0))
    elif which == "summary":
        await _save_fields(db, user_id, summary="")
    else:
        raise ValueError(f"invalid memory file: {which}")
    return await read_snapshot(db, user_id)


async def refresh_from_source(
    db: AsyncSession, user_id: str, source: str,
) -> dict:
    """手动用任意文本材料（例如最近若干条消息）触发一次 LLM 重写。"""
    loaded = await _load_fields(db, user_id, lock=True)
    if loaded is None:
        return {"summary": "", "profile": "", "changed": False}
    summary_cur, profile_raw = loaded
    profile_md = _legacy_profile_to_markdown(profile_raw)
    _, profile_cur = _read_counter(profile_md)

    p_changed, new_profile = await _rewrite_one("profile", profile_cur, source)
    s_changed, new_summary = await _rewrite_one("summary", summary_cur, source)

    await _save_fields(
        db,
        user_id,
        summary=new_summary if s_changed else summary_cur,
        profile=_write_counter(new_profile if p_changed else profile_cur, 0),
    )
    snap = await read_snapshot(db, user_id)
    snap["changed"] = bool(p_changed or s_changed)
    return snap
