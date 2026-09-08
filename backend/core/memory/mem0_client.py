"""Mem0 封装：AsyncMemory 单例（pgvector + DashScope openai-compat）。

用生产级 mem0ai 库替代自写的 mem0_store.py——mem0 提供完整的事实提取、
ADD/UPDATE/DELETE/NOOP 决策、冲突解决、graph memory 等，且经过生产验证。
mem0 的 pgvector provider 启动时自建 memories 表（CREATE EXTENSION + 建表），
所以项目不再需要自管记忆 schema。

主链路调用（mem0 新版 get_all/search 用 filters={"user_id":...} 传用户，
add/delete_all 仍接受顶层 user_id=，delete/update 按 memory_id）：
  get_memory()                                                    -> AsyncMemory 单例（同步初始化）
  await m.add(messages, user_id=..., metadata={"course_id": ...}) 每轮对话后提取记忆，course_id 写入 payload
  await m.search(query, filters={"user_id": ..., "course_id": ...}, top_k=)  注入/检索，按课程隔离
  await m.get_all(filters={"user_id": ...})                       CRUD 列表
  await m.delete(memory_id) / m.update(memory_id, data)           删/改单条

course_id 隔离：写入时把 course_id 存进 payload metadata，检索时 course_id 非空则加入
filters 做 AND 过滤，避免 A 课程聊天产生的记忆串到 B 课程（IM 机器人等无课程概念的调用
方留空 course_id，保持跨课程全局检索）。

增强特性（通过配置开关）：
  - 矛盾检测清理：基于文本相似度规则检测矛盾记忆
  - add 门槛过滤：跳过无意义短消息减少 token 开销
"""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse as up
from datetime import datetime, timezone
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_memory = None  # AsyncMemory | None


def _get_settings():
    """延迟导入 settings 避免循环依赖。"""
    from settings.base import get_settings
    return get_settings()


# persona 槽抽取指令（persona-v1）：mem0 add(infer=True) 的 per-call prompt。
# mem0 2.0.10 的 custom_instr 只追加进 user prompt、system prompt 写死，所以白名单
# 语义靠指令约束 + 抽出后 persona_postfilter 硬过滤兜底（模型仍可能越界）。
# 版本号随指令变更递增——库里旧行无 persona_prompt_version 即词表时代的行。
_PERSONA_INSTRUCTION = """[persona-v1] 从这段师生对话中抽取「该学生的稳定学习者事实」，供跨会话调整讲解方式。

只抽以下类型：
- 讲解/表达偏好（如：喜欢分步讲解、希望讲慢一点、偏好中文术语带英文注释）
- 学习目标与考试节点（如：下周三实验考试、目标是期末 85 分）
- 教材与器材/软件版本（如：用的是第 7 版教材、实验室用 Multisim 14）
- 时间与设备约束（如：每天只能晚上学、手头没有示波器）
- 学生明确要求记住的事

硬性排除（一律不抽）：
- 任何关于掌握程度或对错的结论（如「他不会欧姆定律」「欧姆定律薄弱」）
- 本轮讨论的题目/知识点内容本身
- 情绪状态、一次性事务请求、教师或助手说过的话

输出简体中文。没有符合上述类型的事实时返回空列表。
负例：学生问「能不能再讲一遍欧姆定律」→ 无可抽事实（这是讲解请求，不是偏好）。"""


# 黑名单硬过滤的触发词：抽出文本含「掌握度结论」措辞即丢弃（提示词约束不住时兜底）
_PERSONA_BLACKLIST = ("不会", "薄弱", "掌握", "没懂", "不懂", "薄弱点", "答错", "出错")

# 读侧每槽上限（r1 限权）：persona ≤3 / event ≤2 / 每条 ≤80 字。抽取允许多抽（召回
# 放宽），精度交给读侧限量——与门控「候选集由闭包锁死」同一思路。
_PERSONA_MAX_ITEMS = 3
_EVENT_MAX_ITEMS = 2
_ITEM_MAX_CHARS = 80


def should_skip_user_message(user_message: str) -> bool:
    """门控过滤：判断用户消息是否应被跳过（不写入记忆）。

    与 episodic.record_episode（episode 写入前的门控）共用同一规则，保证
    原始层不进无意义短消息：
    - 命中 skip_patterns（逗号分隔，如"好的"、"嗯"）→ 跳过
    - 长度 < min_length 且 < 5（单字确认等）→ 跳过
    """
    settings = _get_settings()
    user_stripped = (user_message or "").strip()
    if not user_stripped:
        return True

    skip_patterns = settings.mem0.add_skip_patterns.split(",")
    if user_stripped in skip_patterns:
        return True

    if len(user_stripped) < settings.mem0.add_min_length and len(user_stripped) < 5:
        return True

    return False


def _build_config() -> dict:
    """构造 mem0 config：pgvector 指向项目 PG + openai-compat LLM/embedder 指 DashScope。"""
    from settings import get_settings
    DATABASE_URL = get_settings().db.url.get_secret_value()
    DASHSCOPE_API_KEY = get_settings().llm.api_key.get_secret_value()
    DASHSCOPE_BASE_URL = get_settings().llm.base_url
    EMBEDDING_API_KEY = get_settings().embedding.api_key.get_secret_value()
    EMBEDDING_BASE_URL = get_settings().embedding.base_url
    EMBEDDING_MODEL = get_settings().embedding.model
    LIGHTRAG_EMBEDDING_DIM = get_settings().lightrag.embedding_dim
    TEXT_MODEL = get_settings().llm.text_model

    # DATABASE_URL 形如 postgresql+asyncpg://postgres:postgres@postgres:5432/course_agent
    parsed = up.urlparse(DATABASE_URL.replace("+asyncpg", "+psycopg2"))
    return {
        # mem0 OSS 抽取 system prompt（ADDITIVE_EXTRACTION_PROMPT）全是英文 few-shot，
        # 不传 custom_instructions 时 LLM 会把中文对话抽成英文事实。此处强制简体中文。
        "custom_instructions": (
            "CRITICAL: Write every extracted memory in Simplified Chinese (简体中文). "
        
        
        ),
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


async def persona_postfilter(user_id: str, course_id: str, add_resp) -> None:
    """persona 抽取后的硬过滤（plan §8-2 的兜底）：黑名单只在提示词里，mem0 的
    system prompt 覆盖不了，模型仍可能抽出「该生欧姆定律薄弱」写进人设槽。

    对本次 add 返回的事实逐条检查：文本命中掌握度结论措辞（_PERSONA_BLACKLIST）
    即物理删除并 warning（可观测，不静默）。删失败非致命——读侧 header 三禁令仍在。
    """
    try:
        items = add_resp.get("results", []) if isinstance(add_resp, dict) else add_resp
        if not items:
            return
        m = get_memory()
        for it in items:
            if not isinstance(it, dict):
                continue
            text = it.get("memory") or ""
            if not text:
                continue
            if any(w in text for w in _PERSONA_BLACKLIST):
                logger.warning(
                    "[mem0] persona blacklisted (mastery-conclusion leaked), dropped: %s", text
                )
                try:
                    await m.delete(it.get("id"))
                except Exception:
                    logger.warning("[mem0] persona blacklist delete failed id=%s", it.get("id"))
    except Exception as exc:
        logger.warning("[mem0] persona_postfilter failed (non-fatal): %s", exc)


def normalize_results(resp) -> list:
    """mem0 的 get_all/search 返回可能是 ``{"results":[...]}`` 或 ``[...]``，统一成 list。

    core 层（tools.py / build_memory_context / has_any）统一用本函数；api/memory.py 的
    ``_results`` 与本函数同义（api 层保留原样，避免改动范围外代码）。
    """
    if isinstance(resp, dict):
        return resp.get("results", []) or []
    return resp or []


async def _search_slot(
    m, query_text: str, search_filters: dict, top_k: int, threshold: float
) -> list:
    """单槽检索（带 threshold 兼容降级），返回 normalize 后的 list。"""
    try:
        if threshold > 0:
            results = await m.search(
                query_text, filters=search_filters, top_k=top_k, threshold=threshold
            )
        else:
            results = await m.search(query_text, filters=search_filters, top_k=top_k)
    except TypeError:
        # mem0 版本不接受 threshold 关键字 → 降级最小参数重试
        logger.warning(
            "[mem0] search 不接受 threshold 关键字，降级为最小参数重试", exc_info=True
        )
        results = await m.search(query_text, filters=search_filters, top_k=top_k)
    return normalize_results(results)


async def _rerank_slot_if_needed(query_text: str, slot: list, limit: int) -> list:
    """槽内精排（P0）：候选数超截断上限才触发，None/异常一律降级回向量原序。

    精排基建复用 core/rag/rerank.py 的 build_rerank_fn()（qwen3-rerank，契约只要求
    docs 各含 content 字段，与 RAG 共用凭证/超参）。放在硬过滤（主题/门控/矛盾清理）
    之后、截断之前调用——rank 不浪费在注定被滤掉的行上，过滤后 ≤limit 还能省一次
    外部 API 往返（本函数挂在每轮对话同步热路径上，延迟敏感）。
    """
    if len(slot) <= limit:
        return slot
    try:
        from core.rag.rerank import build_rerank_fn  # 延迟导入，与 _get_settings 同风格

        rerank_fn = build_rerank_fn()  # 开关关（mem0/rerank 两级）或无 key → None，行为不变
        if rerank_fn is None:
            return slot
        ranked = await rerank_fn(
            query_text, [{**r, "content": r.get("memory") or ""} for r in slot], top_n=limit
        )
        logger.info(
            "[mem0] slot_reranked candidates=%d limit=%d", len(slot), limit
        )
        return ranked
    except Exception as exc:
        logger.warning("[mem0] slot rerank failed, fallback to vector order: %s", exc)
        return slot


async def build_memory_context(
    user_id: str, query_text: str, *, course_id: str = "", top_k: int = 8, max_chars: int = 1500,
    matched_topic: str = "", gate_evidence_topic: str | None = None,
) -> str:
    """注入对话：按 persona/event 两槽检索该用户记忆，拼成两段文本。无相关则空串。

    读侧限权（两路分工）：第一路（本函数）只答「本主题的历史与表达偏好」，跨主题
    未问及缺口归第二路（proactive 门控）独占。限权靠数据流不靠模型自觉：
    - 每槽独立上限：persona ≤3 条、event ≤2 条、每条 ≤80 字（合计仍受 max_chars 裁剪）
    - matched_topic 非空时 event 槽只注入与本轮命中主题相同的事件（metadata.topic_ids）；
      matched_topic 空但 course_id 非空（问句没命中课程主题）→ event 槽留空，与门控
      「没命中就沉默」对齐；两者都空（IM 无课程概念）才保留旧的全量注入行为
    - gate_evidence_topic 非空（第二路本轮已开口）时，同主题 event 行被剔除——
      同一件事（C2 复发场景）不在两路里说两遍
    - header 三条禁令 + 事件段免责句（不得当掌握度结论/不得展开未问主题/无关时忽略）

    course_id 非空时按课程隔离检索，避免 A 课程聊天产生的记忆（如"正在学 XX 实验课"）
    串到 B 课程的对话里；留空则退化为跨课程全局检索（IM 机器人等无课程概念场景用）。
    """
    if not user_id or not (query_text or "").strip():
        logger.debug("[mem0] build_memory_context skipped: empty user_id or query")
        return ""
    try:
        m = get_memory()
        settings = _get_settings()

        logger.info(
            "[mem0] build_memory_context SEARCH user_id=%s query=%s top_k=%d",
            user_id, query_text[:100], top_k
        )
        threshold = settings.mem0.search_threshold
        base_filters: dict = {"user_id": user_id}
        if course_id:
            base_filters["course_id"] = course_id
        persona = await _search_slot(
            m, query_text, {**base_filters, "kind": "persona"}, top_k, threshold
        )
        events = await _search_slot(
            m, query_text, {**base_filters, "kind": "event"}, top_k, threshold
        )
        results = persona + events
    except Exception as exc:
        logger.warning("[mem0] build_memory_context SEARCH_FAILED user_id=%s error=%s", user_id, exc)
        return ""

    # 矛盾检测清理（如果启用）：清理结果必须接回 persona/events——真正注入用的是这两份
    # 原始槽列表，只清 results 不清槽等于论文 §3.6 读侧矛盾抑制对注入是空操作。
    if settings.mem0.conflict_detect_enabled and len(results) >= 2:
        cleaned = await _detect_and_clean_conflicts(user_id, results, settings)
        if len(cleaned) != len(results):
            keep_ids = {r.get("id") for r in cleaned}
            persona = [r for r in persona if r.get("id") in keep_ids]
            events = [r for r in events if r.get("id") in keep_ids]

    # event 槽按主题结构过滤：只留本轮命中主题的事件（两路分工——跨主题历史交给
    # 门控），再剔除门控本轮已开口的证据主题（同主题不说两遍）。
    # - matched_topic 非空：按命中主题过滤
    # - matched_topic 空但 course_id 非空（有课程、问句没命中主题）：event 槽留空
    #   ——门控这轮已决定不提跨主题学情，全量注入等于另一条路把话又提了
    # - 两者都空（IM 无课程概念）：保留旧的全量注入行为
    if matched_topic:
        events = [
            r for r in events
            if matched_topic in ((r.get("metadata") or {}).get("topic_ids") or [])
        ]
    elif course_id:
        events = []
    if gate_evidence_topic:
        events = [
            r for r in events
            if gate_evidence_topic not in ((r.get("metadata") or {}).get("topic_ids") or [])
        ]

    # P0 读侧精排：所有硬过滤之后、截断之前——两槽各自在候选数超上限时用 qwen3-rerank
    # 重排（向量噪声可能把相关记忆挤到截断线外）。开关关/无 key/失败均降级原序。
    if settings.mem0.rerank_enabled:
        persona, events = await asyncio.gather(
            _rerank_slot_if_needed(query_text, persona, _PERSONA_MAX_ITEMS),
            _rerank_slot_if_needed(query_text, events, _EVENT_MAX_ITEMS),
        )

    persona_items = [r.get("memory") for r in persona if r.get("memory")][:_PERSONA_MAX_ITEMS]
    event_items = [r.get("memory") for r in events if r.get("memory")][:_EVENT_MAX_ITEMS]
    persona_items = [t[:_ITEM_MAX_CHARS] for t in persona_items]
    event_items = [t[:_ITEM_MAX_CHARS] for t in event_items]
    if not persona_items and not event_items:
        logger.info("[mem0] build_memory_context NO_RESULTS user_id=%s", user_id)
        return ""
    # 拼两段标题（## 学习画像 / ## 相关学习事件），无内容的槽整段省略
    parts: list[str] = []
    if persona_items:
        parts.append("## 学习画像\n" + "\n".join(f"- {t}" for t in persona_items))
    if event_items:
        parts.append(
            "## 相关学习事件（以下仅为历史互动记录，不代表当前掌握程度）\n"
            + "\n".join(f"- {t}" for t in event_items)
        )
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    logger.info(
        "[mem0] build_memory_context FOUND user_id=%s persona=%d event=%d total_chars=%d",
        user_id, len(persona_items), len(event_items), len(text)
    )
    return (
        "## 学生个人信息（跨会话，仅用于调整讲解方式）\n"
        "- 只可用于决定讲解风格、称呼、进度与举例，不得作为该生掌握程度的结论\n"
        "- 不得据此主动展开学生本轮未问的知识点（是否提醒薄弱点由「学情提示」段决定）\n"
        "- 与本轮问题无关时忽略，不要在回答里复述这些信息\n"
        f"{text}"
    )


# 矛盾仲裁标签（mem0 官方 ADD/UPDATE/DELETE/NOOP 决策口径的读侧版）：前三个=旧记忆
# 该让位，SUPPLEMENT=互为补充都保留。无法命中标签一律按 SUPPLEMENT 处理（保守不丢条）。
_ARBITRATION_LABELS = ("UPDATE", "CONTRADICT", "SUPPLEMENT", "DUPLICATE")
_ARBITRATION_PROMPT = """你是学生记忆库的矛盾仲裁器。给定同一学生的两条记忆（一旧一新），判断二者关系，只输出以下四个标签之一，不要输出其他内容：

- UPDATE：新记忆更新了旧记忆（旧信息已过时）
- CONTRADICT：两条语义矛盾（如偏好反转）
- DUPLICATE：两条表达同一件事
- SUPPLEMENT：两条互为补充、各有价值，都应保留

前三个标签意味着旧记忆应被剔除；无法确定时输出 SUPPLEMENT。"""


def _row_kind(r: dict) -> str:
    """行 kind：真实行写在 payload metadata（consolidation/tools 的 add 写入），测试
    fixture 放顶层——metadata 优先、顶层兜底，两套都能取到。"""
    return (r.get("metadata") or {}).get("kind") or r.get("kind") or ""


async def _llm_arbitrate_pair(old_text: str, old_dt: datetime, new_text: str, new_dt: datetime) -> str:
    """灰区对的 LLM 四选一仲裁，返回 _ARBITRATION_LABELS 之一。

    temperature=0 + 短输出；异常向上抛由调用方降级。输出里匹配不到合法标签时按
    SUPPLEMENT（保留）——仲裁器说不上来就别替学生扔记忆。
    """
    from core.llm.llm import chat_complete  # 延迟导入：热路径仅灰区对存在时才触达

    out = await chat_complete(
        _ARBITRATION_PROMPT,
        [],
        f"旧记忆（{old_dt.date()}）：{old_text}\n新记忆（{new_dt.date()}）：{new_text}",
        temperature=0.0,
        max_tokens=16,
        langsmith_name="mem0_conflict_arbitration",
    )
    m = re.search(r"\b(UPDATE|CONTRADICT|SUPPLEMENT|DUPLICATE)\b", (out or "").upper())
    return m.group(1) if m else "SUPPLEMENT"


async def _detect_and_clean_conflicts(user_id: str, results: list[dict], settings) -> list[dict]:
    """矛盾检测（两级）：规则粗筛 + 灰区 LLM 仲裁（可配置，默认关）。

    第一级（规则，零成本）：文本相似度 > conflict_similarity_threshold（默认 0.85）
    且时间差 > min_days_gap（默认 7 天）→ 保留较新的，剔除旧的。处理明显重复/字面更新。

    第二级（LLM 仲裁，conflict_llm_arbitration_enabled 开启时）：同 kind、相似度落在
    [conflict_llm_sim_floor, threshold) 灰区的对——语义矛盾但措辞不同（如偏好反转），
    difflib 抓不住——逐对调 LLM 做 UPDATE/CONTRADICT/SUPPLEMENT/DUPLICATE 四选一；
    前三者剔除旧条。LLM 失败/输出非法 → 该对按 SUPPLEMENT 保留（与规则级灰区不结论
    的原行为一致）。仅影响本次注入，不物理删除。

    Args:
        user_id: 用户 ID
        results: search 返回的记忆列表
        settings: 配置对象

    Returns:
        过滤后的记忆列表（排除矛盾的旧记忆）
    """
    threshold = settings.mem0.conflict_similarity_threshold
    min_days_gap = settings.mem0.conflict_min_days_gap
    arbitration_enabled = settings.mem0.conflict_llm_arbitration_enabled
    sim_floor = settings.mem0.conflict_llm_sim_floor

    # 需要排除的记忆 ID
    exclude_ids: set[str] = set()
    # 灰区对（第二级仲裁候选）：(r1, r2, dt1, dt2)
    grayzone: list[tuple[dict, dict, datetime, datetime]] = []

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
            elif (
                arbitration_enabled
                and similarity >= sim_floor
                and _row_kind(r1) and _row_kind(r1) == _row_kind(r2)
            ):
                # 灰区对：同 kind、相似度在 [floor, threshold)——语义矛盾但措辞不同
                # （如偏好反转）时 difflib 抓不住，收集起来交给第二级 LLM 仲裁。
                grayzone.append((r1, r2, dt1, dt2))

    # 第二级：灰区对逐个 LLM 仲裁（串行——每轮灰区对通常 0~3 个，不值得并发复杂度）。
    # 失败/非法输出 → 该对按 SUPPLEMENT 保留，与规则级灰区不结论的原行为一致。
    if arbitration_enabled:
        for r1, r2, dt1, dt2 in grayzone:
            old, old_dt, new, new_dt = (
                (r1, dt1, r2, dt2) if dt1 <= dt2 else (r2, dt2, r1, dt1)
            )
            if old.get("id") in exclude_ids or new.get("id") in exclude_ids:
                continue  # 规则级已剔除一侧，无需重复仲裁
            try:
                label = await _llm_arbitrate_pair(
                    old.get("memory", ""), old_dt, new.get("memory", ""), new_dt
                )
            except Exception as exc:
                logger.warning("[mem0] conflict LLM arbitration failed (keep both): %s", exc)
                continue
            if label != "SUPPLEMENT":
                exclude_ids.add(old.get("id"))
                logger.info(
                    "[mem0] conflict_llm_arbitrated user_id=%s label=%s keep=%s exclude=%s",
                    user_id, label, new.get("id"), old.get("id"),
                )

    # 返回过滤后的结果
    if exclude_ids:
        filtered = [r for r in results if r.get("id") not in exclude_ids]
        logger.info("[mem0] conflict_cleaned user_id=%s original=%d filtered=%d excluded=%d",
                    user_id, len(results), len(filtered), len(exclude_ids))
        return filtered
    return results


async def has_any(user_id: str, course_id: str = "") -> bool:
    """用户是否已有任何记忆（chat 入口的可观测性日志用）。"""
    if not user_id:
        return False
    try:
        m = get_memory()
        filters: dict = {"user_id": user_id}
        if course_id:
            filters["course_id"] = course_id
        results = await m.get_all(filters=filters, top_k=1)
        items = normalize_results(results)
        has = bool(items)
        logger.debug("[mem0] has_any user_id=%s has_memory=%s", user_id, has)
        return has
    except Exception:
        return False
