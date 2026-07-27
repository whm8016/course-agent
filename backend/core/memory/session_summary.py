"""Session L2 摘要管理器。

核心功能：
1. 判断是否需要压缩（maybe_compress）
2. 调用 LLM 压缩旧消息
3. 增量更新摘要（append 模式）

触发条件（任一满足）：
1. 消息数 > WINDOW_SIZE + BUFFER
2. 距上次压缩已过 COMPRESS_INTERVAL 轮

架构位置：
    L1: ContextBuilder 窗口内原文（最近 N 轮）
    L2: Session Summary（早期对话摘要）← 本模块
    L3: mem0 + graph_memory（跨 session 事实和学习图谱）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from settings import get_settings
TEXT_MODEL = get_settings().llm.text_model
from core.db.database import Session, Message
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

# M-11：per-session 压缩锁。main.py 每次 CAPABILITY_COMPLETE 都 create_task 触发
# _maybe_compress_summary，同一 session 多轮 turn 并发触发会同时读 session.summary、各调
# LLM、各自 commit 写 summary_up_to_msg_id → 后者覆盖前者、增量丢失、游标错乱。
# 单事件循环下 dict.setdefault 是原子的，锁按 session_id 隔离；maybe_compress 入口非阻塞
# 探测，已有压缩在进行则本轮让路（下次 turn 再压），避免并发覆盖。
_compress_locks: dict[str, asyncio.Lock] = {}


def _get_compress_lock(session_id: str) -> asyncio.Lock:
    """返回（按需创建）某 session 的压缩锁。单事件循环下 setdefault 原子。"""
    return _compress_locks.setdefault(session_id, asyncio.Lock())

_COMPRESS_PROMPT = """你是对话摘要助手。将下面的对话内容渐进式整合到已有摘要中，输出一份完整的更新摘要。

规则：
1. 保留已有摘要中仍然有效的信息，去除已过时或被纠正的内容
2. 将新对话中的关键信息按时间顺序追加
3. 控制总长度在 300-500 字
4. 使用以下结构输出（如果某节无内容则写"无"）

## 会话主题
（本次会话讨论了哪些主题/领域，按时间顺序列出）

## 关键结论与决定
（双方达成的共识、得出的结论、做出的决定）

## 未解决的问题
（仍在讨论中或尚未回答的问题）

## 约定与后续
（任何关于后续行动的约定，如"下次继续讨论 XX"）

---

已有的早期摘要：
{existing_summary}

---

新增对话：
{new_messages}

---

请输出整合后的完整摘要："""


# ── 结构化 JSON 摘要（P0-B）───────────────────────────────────────────────────
# session.summary 存 JSON 字符串（结构化）；get_summary 负责把它格式化成可读文本注入
# 对话（注入点 chat.py 零改动）。LLM 只抽【新增】内容，已有摘要由代码 merge（去重追加），
# 不交给 LLM 融合——避免 LLM 重写时吞掉旧的关键信息。JSON 解析失败重试 temp=0，再失败
# 降级旧文本逻辑（_COMPRESS_PROMPT）。
# 调研依据：RAPTOR（arXiv:2401.18059）结构化分层摘要、MemGPT（arXiv:2310.08560）结构化
# 记忆块写入而非自由文本；RECOMP（arXiv:2310.04408）selective augmentation（空数组=不相关）。
_STRUCTURED_COMPRESS_PROMPT = """你是对话摘要助手。从【新增对话】中提取结构化信息，输出 JSON。

已有摘要（仅供参考、避免重复抽取，不要原样复制）：
{existing_summary}

新增对话：
{new_messages}

输出以下 JSON 结构（直接输出 JSON，不要 markdown 围栏）：
{{
  "topics": ["本次新增讨论的主题"],
  "decisions": ["本次新增达成的结论或决定"],
  "facts": ["本次确认的具体事实（学生偏好、知识水平、数据等，须可验证）"],
  "open_questions": ["本次仍未解决的问题"],
  "action_items": ["本次约定的后续行动"]
}}

规则：
- 每个数组最多 5 项，每项不超过 50 字
- 只抽取【新增对话】里的新信息，已有摘要仅供参考（代码会去重，重复无害）
- 某类无内容则写空数组 []
- facts 只保留可验证的具体信息，不要笼统描述
"""

_MAX_MSG_CHARS = 2000  # 单条消息喂 LLM 的上限（旧 500 → 2000，避免硬截丢信息）
_MAX_ITEMS_PER_LIST = 5  # 每个 JSON 数组上限（合并后超限丢弃最旧，保时间序）
_SUMMARY_KEYS = ("topics", "decisions", "facts", "open_questions", "action_items")


def _parse_structured(summary_str: str) -> dict | None:
    """解析 session.summary 为 dict。非合法 JSON（旧文本/损坏/空）返回 None。"""
    if not summary_str:
        return None
    try:
        d = json.loads(summary_str)
    except (ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def _parse_json_loose(raw: str) -> dict | None:
    """宽松解析 LLM 输出的 JSON：剥 markdown 围栏、提取首个 {...}，失败返回 None。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            d = json.loads(text[start:end + 1])
            return d if isinstance(d, dict) else None
        except ValueError:
            return None
    return None


def _normalize_item(s: str) -> str:
    """去重归一化：strip + lower + 去末尾常见标点。"""
    return (s or "").strip().lower().rstrip("。.；;，,、:：")


def _merge_structured(existing: dict | None, new: dict) -> dict:
    """代码合并：existing + new 各数组去重追加，每类保留最新 N 项（超限丢最旧）。

    LLM 只抽新内容，已有摘要靠代码 merge——不依赖 LLM 融合，避免吞旧内容。去重按归一化
    文本精确匹配；new 项追加在 existing 之后（保时间序）。
    """
    existing = existing or {}
    merged: dict[str, list[str]] = {}
    for key in _SUMMARY_KEYS:
        old = existing.get(key) if isinstance(existing.get(key), list) else []
        newv = new.get(key) if isinstance(new.get(key), list) else []
        seen: set[str] = set()
        combined: list[str] = []
        for item in [*old, *newv]:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item:
                continue
            norm = _normalize_item(item)
            if norm in seen:
                continue
            seen.add(norm)
            combined.append(item)
        merged[key] = combined[-_MAX_ITEMS_PER_LIST:]
    return merged


def _format_structured(d: dict) -> str:
    """dict → 可读 markdown 文本（注入对话用，与旧自由文本格式相近）。空摘要返回空串。"""
    def _section(title: str, key: str) -> str:
        items = d.get(key) if isinstance(d.get(key), list) else []
        if not items:
            return ""
        body = "\n".join(f"- {it}" for it in items)
        return f"## {title}\n{body}\n\n"

    parts = [
        _section("会话主题", "topics"),
        _section("关键结论与决定", "decisions"),
        _section("确认的事实", "facts"),
        _section("未解决的问题", "open_questions"),
        _section("约定与后续", "action_items"),
    ]
    return "".join(parts).rstrip()


class SessionSummaryManager:
    """Session L2 摘要管理器。"""

    def __init__(
        self,
        window_size: int = 10,
        buffer_size: int = 2,
        compress_interval: int = 5,
    ):
        """初始化摘要管理器。

        Args:
            window_size: L1 窗口大小（轮）
            buffer_size: 超出窗口多少条才触发压缩
            compress_interval: 每隔 N 轮才重新压缩
        """
        self._window_size = window_size
        self._buffer_size = buffer_size
        self._compress_interval = compress_interval

    async def maybe_compress(
        self,
        db: AsyncSession,
        session_id: str,
    ) -> bool:
        """判断是否需要压缩，如需要则执行压缩。

        Args:
            db: 数据库会话
            session_id: 会话 ID

        Returns:
            是否执行了压缩
        """
        # M-11：per-session 并发压缩防护。同一 session 已有压缩在进行时，本轮让路
        # （下次 turn 再压），避免两个并发 maybe_compress 同时读旧 summary、各自调 LLM、
        # 各自 commit 导致后者覆盖前者、summary_up_to_msg_id 游标错乱、增量丢失。
        # 非阻塞探测：locked() 检查与 async with 之间无 await，单事件循环下无 TOCTOU。
        lock = _get_compress_lock(session_id)
        if lock.locked():
            logger.debug("[L2] compress already in progress session=%s; skip this round", session_id)
            return False

        async with lock:
            return await self._maybe_compress_locked(db, session_id)

    async def _maybe_compress_locked(
        self,
        db: AsyncSession,
        session_id: str,
    ) -> bool:
        """实际的压缩逻辑（已持 per-session 锁，由 maybe_compress 调用）。"""
        # 1. 获取 session 和消息列表
        session = await db.get(Session, session_id)
        if not session:
            logger.warning("[L2] session not found: %s", session_id)
            return False

        # 2. 获取所有消息（按时间排序）
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        messages = list(result.scalars().all())

        total_msgs = len(messages)
        threshold = self._window_size + self._buffer_size

        # 3. 判断是否需要压缩
        if total_msgs <= threshold:
            logger.debug(
                "[L2] no need to compress session=%s msgs=%d threshold=%d",
                session_id, total_msgs, threshold
            )
            return False

        # 4. 找出需要压缩的消息（窗口之外的旧消息）
        #    窗口保留最近 window_size 轮（user + assistant 配对）
        window_msg_count = self._window_size * 2  # user + assistant

        # 如果消息数不足以留出窗口，不压缩
        if total_msgs <= window_msg_count:
            return False

        messages_to_compress = messages[:-window_msg_count]

        if not messages_to_compress:
            return False

        # 5. 检查是否已有摘要，判断增量压缩还是全量压缩
        existing_summary = session.summary or ""
        last_compressed_id = session.summary_up_to_msg_id

        # 找出新增的需要压缩的消息
        if last_compressed_id:
            # 增量：从 last_compressed_id 之后开始
            try:
                last_idx = next(
                    i for i, m in enumerate(messages)
                    if m.id == last_compressed_id
                )
                new_messages_to_compress = messages[last_idx + 1:-window_msg_count]
            except StopIteration:
                new_messages_to_compress = messages_to_compress
        else:
            new_messages_to_compress = messages_to_compress

        if not new_messages_to_compress:
            logger.debug("[L2] no new messages to compress session=%s", session_id)
            return False

        # 6. 检查压缩频率（避免每轮都压缩）
        #    如果已有摘要，检查距离上次压缩过了多少轮
        if last_compressed_id and len(new_messages_to_compress) < self._compress_interval:
            logger.debug(
                "[L2] skip compress session=%s new_msgs=%d < interval=%d",
                session_id, len(new_messages_to_compress), self._compress_interval
            )
            return False

        # 7. 调用 LLM 压缩
        new_summary = await self._do_compress(
            existing_summary,
            new_messages_to_compress,
        )

        if not new_summary:
            return False

        # 8. 更新 session
        session.summary = new_summary
        session.summary_up_to_msg_id = messages_to_compress[-1].id
        session.summary_updated_at = time.time()
        await db.commit()

        logger.info(
            "[L2] compress complete session=%s total_msgs=%d compressed=%d summary_len=%d",
            session_id, total_msgs, len(new_messages_to_compress), len(new_summary)
        )
        return True

    async def _do_compress(
        self,
        existing_summary: str,
        messages: list[Message],
    ) -> str | None:
        """结构化 JSON 抽取 + 代码增量合并；失败降级旧文本逻辑。

        返回值写入 session.summary：成功=JSON 字符串，降级=自由文本。get_summary 负责
        把 JSON 格式化成可读文本注入对话（注入点零改动）。LLM 只抽【新增】内容，已有摘要
        由 ``_merge_structured`` 代码合并去重——不交给 LLM 融合，避免吞旧内容。
        """
        msg_text = "\n".join(
            f"{m.role}: {m.content[:_MAX_MSG_CHARS]}"
            for m in messages
        )
        existing = _parse_structured(existing_summary)
        existing_ref = _format_structured(existing) if existing else "(无)"

        # 结构化抽取：temp=0.3 首试，解析失败 temp=0 重试一次
        for temperature in (0.3, 0.0):
            prompt = _STRUCTURED_COMPRESS_PROMPT.format(
                existing_summary=existing_ref,
                new_messages=msg_text,
            )
            raw = ""
            try:
                resp = await async_openai_client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=800,
                )
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning("[L2] structured compress temp=%s failed: %s", temperature, e)
            new = _parse_json_loose(raw)
            if new is not None:
                merged = _merge_structured(existing, new)
                if any(merged.get(k) for k in _SUMMARY_KEYS):
                    logger.info(
                        "[L2] structured compress done keys=%s",
                        [k for k in _SUMMARY_KEYS if merged.get(k)],
                    )
                    return json.dumps(merged, ensure_ascii=False)
                logger.warning("[L2] structured compress returned empty json, retry/fallback")

        # 降级：旧自由文本融合逻辑（结构化连续失败时兜底，保证不阻塞）
        logger.warning("[L2] structured compress exhausted, fallback to text summary")
        return await self._do_compress_text(existing_summary, messages)

    async def _do_compress_text(
        self,
        existing_summary: str,
        messages: list[Message],
    ) -> str | None:
        """降级路径：旧自由文本融合（结构化 JSON 抽取连续失败时兜底，保证不阻塞）。"""
        msg_text = "\n".join(
            f"{m.role}: {m.content[:_MAX_MSG_CHARS]}"
            for m in messages
        )
        prompt = _COMPRESS_PROMPT.format(
            existing_summary=existing_summary or "(无)",
            new_messages=msg_text,
        )
        try:
            resp = await async_openai_client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            summary = (resp.choices[0].message.content or "").strip()
            logger.info("[L2] text fallback compress done summary_len=%d", len(summary))
            return summary or None
        except Exception as e:
            logger.warning("[L2] text fallback compress failed: %s", e)
            return None

    async def get_summary(self, db: AsyncSession, session_id: str) -> str:
        """获取 session 的 L2 摘要（注入对话用）。

        session.summary 存 JSON 字符串（结构化）或旧文本（降级）。JSON 自动格式化成可读
        markdown 文本返回；非 JSON 原样返回。注入点（chat.py）拿到的始终是可读文本，零改动。
        """
        session = await db.get(Session, session_id)
        if not session:
            return ""
        raw = session.summary or ""
        structured = _parse_structured(raw)
        if structured is not None:
            return _format_structured(structured)
        return raw


# 全局单例
_summary_manager: SessionSummaryManager | None = None


def get_summary_manager() -> SessionSummaryManager:
    """返回全局 SessionSummaryManager 单例。"""
    global _summary_manager
    if _summary_manager is None:
        # 嵌套配置 settings.summary（config-nested-refactor）：字段 window/buffer/interval。
        # 此前误用扁平 settings.summary_window_size → AttributeError 被 _maybe_compress_summary
        # 静默吞掉，导致 L2 摘要从不生成、不落库、langsmith 也 trace 不到 _do_compress。
        cfg = get_settings().summary
        _summary_manager = SessionSummaryManager(
            window_size=cfg.window_size,
            buffer_size=cfg.buffer_size,
            compress_interval=cfg.compress_interval,
        )
        logger.info(
            "[L2] SessionSummaryManager initialized window=%d buffer=%d interval=%d",
            cfg.window_size, cfg.buffer_size, cfg.compress_interval,
        )
    return _summary_manager


__all__ = [
    "SessionSummaryManager",
    "get_summary_manager",
    "_get_compress_lock",
    "_compress_locks",
]
