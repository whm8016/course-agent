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

import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from settings import get_settings
TEXT_MODEL = get_settings().llm.text_model
from core.db.database import Session, Message
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

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
        """调用 LLM 执行压缩。"""
        # 格式化消息
        msg_text = "\n".join(
            f"{m.role}: {m.content[:500]}"
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
            logger.info("[L2] LLM compress done summary_len=%d", len(summary))
            return summary
        except Exception as e:
            logger.warning("[L2] compress failed: %s", e)
            return None

    async def get_summary(self, db: AsyncSession, session_id: str) -> str:
        """获取 session 的 L2 摘要。"""
        session = await db.get(Session, session_id)
        if not session:
            return ""
        return session.summary or ""


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


__all__ = ["SessionSummaryManager", "get_summary_manager"]
