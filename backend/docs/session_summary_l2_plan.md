# Session L2 摘要层实现方案

## Context

当前系统存在三层记忆架构的缺口：

```
L1: ContextBuilder 窗口内原文（最近 N 轮）  ← 会丢失旧消息
L2: ❌ 缺失                                     ← 需要实现
L3: mem0 跨 session 事实记忆                    ← 只存离散事实，无脉络
```

**问题**：
- L1 按窗口截断，旧消息直接丢弃，长对话前半段脉络消失
- L3 只存离散事实（"用户偏好 X"），不记录连续对话脉络
- LLM 会重复问已讨论过的内容，或忘记之前的结论

**目标**：增加 L2 层，将被窗口丢弃的旧消息压缩为摘要，注入 system prompt。

---

## 最终架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Prompt 组装                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Course System Prompt（课程设定）                             │
│  2. Memory Context（mem0 事实，L3）                              │
│  3. Session Summary（早期对话摘要，L2）← 新增                     │
│  4. Skills Manifest（技能清单）                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Messages 列表（L1）                          │
│  [最近 N 轮原文，由 ContextBuilder 管理]                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 数据模型变更

### 1. Session 表新增字段

**文件**: `backend/core/db/database.py`

```python
class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True)
    course_id = Column(String(64), nullable=False)
    user_id = Column(String(32), nullable=False)
    title = Column(String(256), nullable=False, default="新对话")
    mode = Column(String(32), nullable=False, default="chat")
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)

    # 新增字段 ↓
    summary = Column(Text, nullable=False, default="")  # L2 摘要文本
    summary_up_to_msg_id = Column(String(32), nullable=True)  # 摘要覆盖到哪条消息
    summary_updated_at = Column(Float, nullable=True)  # 摘要最后更新时间

    messages = relationship("Message", back_populates="session")
```

### 2. 数据库迁移

**文件**: `backend/alembic/versions/xxx_add_session_summary.py`

```python
def upgrade():
    op.add_column('sessions', sa.Column('summary', sa.Text(), nullable=False, server_default=''))
    op.add_column('sessions', sa.Column('summary_up_to_msg_id', sa.String(32), nullable=True))
    op.add_column('sessions', sa.Column('summary_updated_at', sa.Float(), nullable=True))

def downgrade():
    op.drop_column('sessions', 'summary_updated_at')
    op.drop_column('sessions', 'summary_up_to_msg_id')
    op.drop_column('sessions', 'summary')
```

---

## 核心组件设计

### 1. SessionSummaryManager

**文件**: `backend/core/memory/session_summary.py`

```python
"""Session L2 摘要管理器。

核心功能：
1. 判断是否需要压缩（maybe_compress）
2. 调用 LLM 压缩旧消息
3. 增量更新摘要（append 模式）

触发条件（任一满足）：
1. 消息数 > WINDOW_SIZE + BUFFER
2. 距上次压缩已过 COMPRESS_INTERVAL 轮
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import Session, Message
from config import TEXT_MODEL
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

# 配置常量（可移至 settings）
WINDOW_SIZE = 10  # L1 窗口大小
BUFFER_SIZE = 2   # 超出窗口多少条才触发压缩
COMPRESS_INTERVAL = 5  # 每隔 N 轮才重新压缩一次

_COMPRESS_PROMPT = """你是对话摘要助手。把下面的师生对话压缩为 300-500 字的摘要。

重点保留：
1. 讨论了哪些话题/知识点（按时间顺序）
2. 学生的核心疑惑和已解决的问题
3. 尚未解决的遗留问题
4. 任何约定（"下次继续讲 XX"）

不要保留具体的解题步骤细节，只保留结论和进展。

---

已有的早期摘要：
{existing_summary}

---

新增对话（需要压缩的部分）：
{new_messages}

---

请输出整合后的完整摘要（包含已有摘要 + 新增部分的压缩）："""


class SessionSummaryManager:
    """Session L2 摘要管理器。"""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        buffer_size: int = BUFFER_SIZE,
        compress_interval: int = COMPRESS_INTERVAL,
    ):
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
            logger.debug("[L2] no need to compress session=%s msgs=%d threshold=%d",
                        session_id, total_msgs, threshold)
            return False

        # 4. 找出需要压缩的消息（窗口之外的旧消息）
        #    窗口保留最近 window_size * 2 条（user + assistant 配对）
        window_pair_count = self._window_size
        window_msg_count = window_pair_count * 2  # user + assistant

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

        # 6. 调用 LLM 压缩
        new_summary = await self._do_compress(
            existing_summary,
            new_messages_to_compress,
        )

        if not new_summary:
            return False

        # 7. 更新 session
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
        from settings.base import get_settings
        settings = get_settings()
        _summary_manager = SessionSummaryManager(
            window_size=settings.summary_window_size,
            buffer_size=settings.summary_buffer_size,
            compress_interval=settings.summary_compress_interval,
        )
    return _summary_manager
```

---

## 调用链路

### 1. 触发时机

**文件**: `backend/main.py`

在 `_on_capability_complete` 中，与 flush_manager 一起触发：

```python
async def _on_capability_complete(event) -> None:
    """CAPABILITY_COMPLETE：批量 enqueue 到 flush_manager + 触发 L2 压缩。"""
    try:
        # 1. mem0 + graph_memory 批量 flush
        from core.memory.flush_manager import get_flush_manager
        flush_mgr = get_flush_manager()
        session_id = getattr(event, "session_id", "") or event.metadata.get("session_id", "") or ""
        await flush_mgr.enqueue(
            user_id=event.user_id,
            session_id=session_id,
            course_id=event.course_id,
            user_msg=event.user_message,
            assistant_msg=event.agent_output,
        )

        # 2. L2 摘要压缩（异步，不阻塞）
        if session_id:
            asyncio.create_task(_maybe_compress_summary(session_id))

    except Exception:
        logger.warning("EventBus: memory enqueue failed", exc_info=True)


async def _maybe_compress_summary(session_id: str) -> None:
    """异步触发 L2 压缩（不阻塞当前 turn）。"""
    try:
        from core.memory.session_summary import get_summary_manager
        from core.db.database import AsyncSessionLocal

        summary_mgr = get_summary_manager()
        async with AsyncSessionLocal() as db:
            await summary_mgr.maybe_compress(db, session_id)
    except Exception as e:
        logger.warning("[L2] maybe_compress failed session=%s error=%s", session_id, e)
```

### 2. 注入 UnifiedContext

**文件**: `backend/core/context.py`

```python
@dataclass
class UnifiedContext:
    # ... 现有字段 ...
    memory_context: str = ""  # L3: mem0 事实
    session_summary: str = ""  # L2: 早期对话摘要 ← 新增
    # ... 其他字段 ...
```

### 3. chat.py 读取摘要

**文件**: `backend/api/chat.py`

```python
# 读 Mem0 记忆（L3）
from core.memory.mem0_client import build_memory_context as _mem_ctx_fn
_mem_ctx = await _mem_ctx_fn(str(user["id"]), message)

# 读 Session Summary（L2）← 新增
_session_summary = ""
if session_id:
    from core.memory.session_summary import get_summary_manager
    from core.db.database import AsyncSessionLocal
    summary_mgr = get_summary_manager()
    async with AsyncSessionLocal() as db:
        _session_summary = await summary_mgr.get_summary(db, session_id)

ctx = UnifiedContext(
    # ... 现有字段 ...
    memory_context=_mem_ctx,  # L3
    session_summary=_session_summary,  # L2 ← 新增
)
```

### 4. System Prompt 注入

**文件**: `backend/core/capabilities/chat_pipeline.py`

在 system prompt 组装处添加：

```python
def _build_system_prompt(self, ctx: UnifiedContext) -> str:
    parts = [course_system_prompt]

    # L3: mem0 事实记忆
    if ctx.memory_context:
        parts.append(ctx.memory_context)

    # L2: 早期对话摘要 ← 新增
    if ctx.session_summary:
        parts.append(
            f"## 本次对话前情摘要（早期对话的压缩，非完整原文）\n{ctx.session_summary}"
        )

    # Skills manifest
    if ctx.skills_manifest:
        parts.append(ctx.skills_manifest)

    return "\n\n".join(parts)
```

---

## 配置项

**文件**: `backend/settings/base.py`

```python
class Settings(BaseSettings):
    # ... 现有配置 ...

    # L2 摘要配置
    summary_window_size: int = 10  # L1 窗口大小（轮）
    summary_buffer_size: int = 2   # 超出窗口多少条才触发压缩
    summary_compress_interval: int = 5  # 每隔 N 轮才重新压缩
```

**.env 新增**:
```env
SUMMARY_WINDOW_SIZE=10
SUMMARY_BUFFER_SIZE=2
SUMMARY_COMPRESS_INTERVAL=5
```

---

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|-----|---------|------|
| `backend/core/db/database.py` | 修改 | Session 模型加 summary 等 3 字段 |
| `backend/alembic/versions/xxx_add_session_summary.py` | 新建 | 数据库迁移 |
| `backend/core/memory/session_summary.py` | 新建 | SessionSummaryManager 核心逻辑 |
| `backend/core/context.py` | 修改 | UnifiedContext 加 session_summary 字段 |
| `backend/api/chat.py` | 修改 | 读取 session.summary 注入 context |
| `backend/core/capabilities/chat_pipeline.py` | 修改 | system prompt 组装时注入摘要 |
| `backend/main.py` | 修改 | _on_capability_complete 触发压缩 |
| `backend/settings/base.py` | 修改 | 加 3 个配置项 |

---

## Token 开销估算

- 触发频率：每 `COMPRESS_INTERVAL` 轮（默认 5 轮）压缩一次
- 单次压缩：~2000-3000 input tokens + ~500 output tokens
- 30 轮对话：压缩 4-5 次 ≈ ~12k tokens
- 对比每轮都调 LLM：30 轮 × 2k = 60k tokens

**节省约 80% token 开销**。

---

## 与现有系统的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    三层记忆架构（完整）                           │
├─────────────────────────────────────────────────────────────────┤
│  L1: ContextBuilder 窗口内原文    → 最近 N 轮完整对话            │
│  L2: Session Summary（新增）      → 早期对话压缩摘要             │
│  L3: mem0 + graph_memory         → 跨 session 事实 + 学习图谱   │
└─────────────────────────────────────────────────────────────────┘

各层职责：
- L1 保留最新上下文（精确但有限）
- L2 保留历史脉络（压缩但有连续性）
- L3 保留长期记忆（事实和学习进度）

互不冲突，可同时工作。
```

---

## 验证标准

1. 消息数 <= threshold 时不触发压缩
2. 压缩后 `session.summary` 非空
3. `summary_up_to_msg_id` 正确记录
4. 下一轮对话时 system prompt 包含摘要
5. 长对话（>30轮）中早期内容仍被"记住"

---

## 实施顺序

1. **Step 1**: Session 模型加字段 + 迁移
2. **Step 2**: 新建 session_summary.py
3. **Step 3**: UnifiedContext 加字段
4. **Step 4**: chat.py 读取摘要
5. **Step 5**: chat_pipeline.py 注入 system prompt
6. **Step 6**: main.py 触发压缩
7. **Step 7**: settings 加配置项
