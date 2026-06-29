"""Mem0 封装：AsyncMemory 单例（pgvector + DashScope openai-compat）。

用生产级 mem0ai 库替代自写的 mem0_store.py——mem0 提供完整的事实提取、
ADD/UPDATE/DELETE/NOOP 决策、冲突解决、graph memory 等，且经过生产验证。
mem0 的 pgvector provider 启动时自建 memories 表（CREATE EXTENSION + 建表），
所以项目不再需要自管记忆 schema。

主链路调用：
  get_memory()                                    -> AsyncMemory 单例（同步初始化）
  await m.add(messages, user_id=...)              每轮对话后提取记忆
  await m.search(query, user_id=..., top_k=)      注入/检索
  await m.get_all(user_id=...) / delete / update  CRUD

增强特性（通过配置开关）：
  - 时间衰减评分：recency_decay_lambda 参数让新记忆得分更高
  - 矛盾检测清理：基于文本相似度规则检测矛盾记忆
  - add 门槛过滤：跳过无意义短消息减少 token 开销
"""
from __future__ import annotations

import logging
import urllib.parse as up
from datetime import datetime, timezone
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_memory = None  # AsyncMemory | None


def _get_settings():
    """延迟导入 settings 避免循环依赖。"""
    from settings.base import get_settings
    return get_settings()


def should_skip_user_message(user_message: str) -> bool:
    """门控过滤：判断用户消息是否应被跳过（不写入记忆）。

    与 add_turn / flush_manager.enqueue 共用同一规则，保证生产者-消费者
    两端过滤口径一致：
    - 命中 skip_patterns（逗号分隔，如"好的"、"嗯"）→ 跳过
    - 长度 < min_length 且 < 5（单字确认等）→ 跳过
    """
    settings = _get_settings()
    user_stripped = (user_message or "").strip()
    if not user_stripped:
        return True

    skip_patterns = settings.mem0_add_skip_patterns.split(",")
    if user_stripped in skip_patterns:
        return True

    if len(user_stripped) < settings.mem0_add_min_length and len(user_stripped) < 5:
        return True

    return False


def _build_config() -> dict:
    """构造 mem0 config：pgvector 指向项目 PG + openai-compat LLM/embedder 指 DashScope。"""
    from config import (
        DATABASE_URL,
        DASHSCOPE_API_KEY,
        DASHSCOPE_BASE_URL,
        EMBEDDING_API_KEY,
        EMBEDDING_BASE_URL,
        EMBEDDING_MODEL,
        LIGHTRAG_EMBEDDING_DIM,
        TEXT_MODEL,
    )

    # DATABASE_URL 形如 postgresql+asyncpg://postgres:postgres@postgres:5432/course_agent
    parsed = up.urlparse(DATABASE_URL.replace("+asyncpg", "+psycopg2"))
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
                "user": parsed.username or "postgres",
                "password": parsed.password or "",
                "dbname": (parsed.path or "/course_agent").lstrip("/"),
                "collection_name": "memories",
                "embedding_model_dims": LIGHTRAG_EMBEDDING_DIM,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": TEXT_MODEL,
                "api_key": DASHSCOPE_API_KEY,
                "openai_base_url": DASHSCOPE_BASE_URL,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBEDDING_MODEL,
                "api_key": EMBEDDING_API_KEY,
                "openai_base_url": EMBEDDING_BASE_URL,
            },
        },
    }


def get_memory():
    """返回 AsyncMemory 单例。

    from_config 是同步的（mem0 Issue #2755），首次调用时连 PG + 建表，建议在 app
    startup 预热一次（lifespan）避免首条对话卡顿。
    """
    global _memory
    if _memory is None:
        from mem0 import AsyncMemory

        _memory = AsyncMemory.from_config(_build_config())
        logger.info("Mem0 AsyncMemory initialized (pgvector + openai-compat LLM)")
    return _memory


async def add_turn(user_id: str, user_message: str, assistant_message: str) -> None:
    """每轮对话后提取记忆（EventBus / worker 调用）。

    增强特性：
    - 消息长度门槛过滤：跳过无意义短消息（如"好的"、"嗯"）
    """
    if not user_id:
        logger.warning("[mem0] add_turn skipped: empty user_id")
        return

    # 门槛过滤：跳过无意义短消息（与 flush_manager.enqueue 共用 should_skip_user_message）
    user_stripped = user_message.strip()
    if should_skip_user_message(user_message):
        logger.info("[mem0] add_turn SKIPPED user_id=%s reason=gate msg=%s", user_id, user_stripped[:20])
        return

    m = get_memory()
    logger.info(
        "[mem0] add_turn START user_id=%s user_msg_len=%d assistant_msg_len=%d",
        user_id, len(user_message), len(assistant_message)
    )
    logger.debug("[mem0] add_turn INPUT user_id=%s user_msg=%s assistant_msg=%s",
                 user_id, user_message[:200], assistant_message[:200])
    result = await m.add(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        user_id=user_id,
    )
    # mem0 add 返回可能是 list 或 dict，记录提取/存储的结果
    if result:
        items = result if isinstance(result, list) else result.get("results", [])
        logger.info(
            "[mem0] add_turn COMPLETE user_id=%s stored_count=%d items=%s",
            user_id, len(items) if items else 0,
            [i.get("memory", "")[:50] if isinstance(i, dict) else str(i)[:50] for i in (items or [])]
        )
    else:
        logger.info("[mem0] add_turn COMPLETE user_id=%s stored_count=0 (no new facts)", user_id)


async def build_memory_context(user_id: str, query_text: str, *, top_k: int = 6, max_chars: int = 2000) -> str:
    """注入对话：用 query_text 语义检索该用户相关记忆，拼成文本。无相关则空串。

    增强特性（通过配置开关）：
    - 时间衰减评分：新记忆得分更高（recency_decay_lambda）
    - 矛盾检测清理：基于文本相似度规则检测并排除矛盾记忆
    """
    if not user_id or not (query_text or "").strip():
        logger.debug("[mem0] build_memory_context skipped: empty user_id or query")
        return ""
    try:
        m = get_memory()
        settings = _get_settings()

        # 时间衰减参数（从配置读取）
        decay_lambda = 0.0
        if settings.mem0_time_decay_enabled:
            decay_lambda = settings.mem0_time_decay_lambda

        logger.info(
            "[mem0] build_memory_context SEARCH user_id=%s query=%s top_k=%d decay_lambda=%.4f",
            user_id, query_text[:100], top_k, decay_lambda
        )
        results = await m.search(
            query_text,
            filters={"user_id": user_id},
            top_k=top_k,
            recency_decay_lambda=decay_lambda,
        )
        results = results.get("results", []) if isinstance(results, dict) else (results or [])
    except Exception as exc:
        logger.warning("[mem0] build_memory_context SEARCH_FAILED user_id=%s error=%s", user_id, exc)
        return ""

    # 矛盾检测清理（如果启用）
    if settings.mem0_conflict_detect_enabled and len(results) >= 2:
        results = await _detect_and_clean_conflicts(user_id, results, settings)

    items = [r.get("memory") for r in (results or []) if r.get("memory")]
    if not items:
        logger.info("[mem0] build_memory_context NO_RESULTS user_id=%s", user_id)
        return ""
    text = "\n".join(f"- {t}" for t in items)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    logger.info(
        "[mem0] build_memory_context FOUND user_id=%s count=%d total_chars=%d memories=%s",
        user_id, len(items), len(text), [t[:50] for t in items]
    )
    return f"## 用户记忆（仅在直接相关时参考，不要硬塞进答案）\n{text}"


async def _detect_and_clean_conflicts(user_id: str, results: list[dict], settings) -> list[dict]:
    """基于文本相似度规则检测矛盾记忆，保留较新的那条。

    判定条件：
    1. 两条记忆的文本相似度 > threshold（默认 0.85）
    2. 两条记忆的时间差 > min_days_gap（默认 7 天）

    满足条件时，保留 created_at 更新的那条，排除旧的那条。
    零 LLM token 开销，纯 difflib.SequenceMatcher 计算。

    Args:
        user_id: 用户 ID
        results: search 返回的记忆列表
        settings: 配置对象

    Returns:
        过滤后的记忆列表（排除矛盾的旧记忆）
    """
    threshold = settings.mem0_conflict_similarity_threshold
    min_days_gap = settings.mem0_conflict_min_days_gap

    # 需要排除的记忆 ID
    exclude_ids: set[str] = set()

    # 两两比较（最多 C(top_k, 2) 对）
    for i, r1 in enumerate(results):
        mem1_id = r1.get("id")
        if mem1_id in exclude_ids:
            continue
        mem1_text = r1.get("memory", "")
        mem1_created = r1.get("created_at")

        if not mem1_text or not mem1_created:
            continue

        # 解析时间
        try:
            if isinstance(mem1_created, str):
                dt1 = datetime.fromisoformat(mem1_created.replace("Z", "+00:00"))
            elif isinstance(mem1_created, datetime):
                dt1 = mem1_created
            else:
                continue
            if dt1.tzinfo is None:
                dt1 = dt1.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        for j, r2 in enumerate(results[i + 1:], start=i + 1):
            mem2_id = r2.get("id")
            if mem2_id in exclude_ids:
                continue
            mem2_text = r2.get("memory", "")
            mem2_created = r2.get("created_at")

            if not mem2_text or not mem2_created:
                continue

            # 解析时间
            try:
                if isinstance(mem2_created, str):
                    dt2 = datetime.fromisoformat(mem2_created.replace("Z", "+00:00"))
                elif isinstance(mem2_created, datetime):
                    dt2 = mem2_created
                else:
                    continue
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            # 计算文本相似度（SequenceMatcher）
            similarity = SequenceMatcher(None, mem1_text, mem2_text).ratio()

            if similarity > threshold:
                # 检查时间差
                days_gap = abs((dt1 - dt2).total_seconds()) / 86400.0
                if days_gap > min_days_gap:
                    # 保留较新的，排除较旧的
                    if dt1 > dt2:
                        exclude_ids.add(mem2_id)
                        logger.info(
                            "[mem0] conflict_detected user_id=%s keep=%s (%s) exclude=%s (%s) similarity=%.2f gap=%.1f days",
                            user_id, mem1_id, dt1.date(), mem2_id, dt2.date(), similarity, days_gap
                        )
                    else:
                        exclude_ids.add(mem1_id)
                        logger.info(
                            "[mem0] conflict_detected user_id=%s keep=%s (%s) exclude=%s (%s) similarity=%.2f gap=%.1f days",
                            user_id, mem2_id, dt2.date(), mem1_id, dt1.date(), similarity, days_gap
                        )

    # 返回过滤后的结果
    if exclude_ids:
        filtered = [r for r in results if r.get("id") not in exclude_ids]
        logger.info("[mem0] conflict_cleaned user_id=%s original=%d filtered=%d excluded=%d",
                    user_id, len(results), len(filtered), len(exclude_ids))
        return filtered
    return results


async def get_all_text(user_id: str) -> str:
    """read_memory 工具：返回该用户全部记忆文本。"""
    if not user_id:
        logger.warning("[mem0] get_all_text skipped: empty user_id")
        return "(未登录)"
    m = get_memory()
    logger.info("[mem0] get_all_text FETCH user_id=%s", user_id)
    results = await m.get_all(user_id=user_id, top_k=200)
    rows = [r for r in (results.get("results", []) if isinstance(results, dict) else (results or [])) if r.get("memory")]
    if not rows:
        logger.info("[mem0] get_all_text NO_RESULTS user_id=%s", user_id)
        return "(暂无记忆)"
    logger.info(
        "[mem0] get_all_text FOUND user_id=%s count=%d memories=%s",
        user_id, len(rows), [r.get("memory", "")[:50] for r in rows[:5]]
    )
    # 带 id，供 write_memory 的 edit 拿到 target_id
    return "\n".join(f"- [id={r.get('id')}] {r.get('memory', '')}" for r in rows)


async def has_any(user_id: str) -> bool:
    """用户是否已有任何记忆（决定是否挂 read_memory 工具）。"""
    if not user_id:
        return False
    try:
        m = get_memory()
        results = await m.get_all(user_id=user_id, top_k=1)
        items = results.get("results", []) if isinstance(results, dict) else (results or [])
        has = bool(items)
        logger.debug("[mem0] has_any user_id=%s has_memory=%s", user_id, has)
        return has
    except Exception:
        return False
