"""写侧零 LLM 闭集匹配：对话文本 → 课程主题命中 + 规则行为信号。

替代 consolidate 里的 LLM 开放抽取（extract_knowledge）：主题命中靠 embedding 最近邻
（课程主题图是教师审计过的闭集，抽象已由建库期提供，运行时生成式抽取是纯浪费——
Zero-Mem 的论证结构）；行为信号（困惑/纠正/偏好/重问）靠关键词规则，零 LLM。

只有 should_route_error_llm 判「值得抽错题」时才回退 LLM（SAGE 的不确定带路由：
判据是行为证据而非 embedding 密度，因为要抽的是错误模式不是新事实）。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 行为信号关键词（中文对话规则）。首版保守：宁可漏检不可误报——误报会触发多余的
# LLM 错题抽取，而漏检只损失一次 error_patterns 观测，cron 兜底窗口还能补。
_CONFUSION_PATTERNS = (
    "不懂", "不明白", "没看懂", "看不懂", "听不懂", "不理解", "没理解", "什么意思",
    "为什么不对", "哪里错了", "还是不会", "还是不懂", "卡住了", "做不出来", "不知道怎么",
)
_CORRECTION_PATTERNS = (
    "不对，应该是", "不是，是", "错了，", "说错了", "应该是", "你搞错了", "你记错了",
)
_PREFERENCE_PATTERNS = (
    "我喜欢", "我偏好", "我希望你", "请用", "我习惯", "我更喜欢", "能不能",
)
_NEGATIVE_ACK_PATTERNS = ("不是这个", "没用", "没帮助", "还是错", "又错了")
_AFFIRM_PATTERNS = ("对的", "没错", "正确", "正是", "就是这个", "懂了", "明白了", "学会了")

# 信号 → mastery delta（risk 上升=薄弱加重；mastery 上升=掌握改善）
_RISK_DELTA = {"confusion": 0.15, "correction": 0.2, "negative_ack": 0.15, "repeat": 0.1}
_MASTERY_DELTA = {"affirm": 0.1}


def _hit_any(text: str, patterns: tuple[str, ...]) -> str | None:
    """返回 text 命中的第一个模式（无则 None）。逐 pattern 而非正则拼接，便于命中词回日志。"""
    for p in patterns:
        if p in text:
            return p
    return None


def infer_signal(
    user_msg: str, assistant_msg: str, *, history_hint: dict | None = None
) -> dict:
    """纯函数：从一轮对话的文本 + 历史提示推断行为信号（零 LLM）。

    Args:
        user_msg / assistant_msg: 本轮原始对话。
        history_hint: {"topic_repeat_count": 该主题此前已观测次数}——同主题重复提问
            是比文本更强的困惑证据（问过还问 = 上次没讲透），来自 mastery 行的
            observation_count（本轮写入前），无行则 0。

    Returns:
        {"signals": [...], "risk_delta": float, "mastery_delta": float,
         "topic_repeat_count": int}
        signals 语义标签（confusion/correction/negative_ack/preference/affirm/repeat），
        delta 已按规则合并，可直接喂 append_mastery_from_topics。
    """
    user = user_msg or ""
    signals: list[str] = []

    if _hit_any(user, _CONFUSION_PATTERNS):
        signals.append("confusion")
    # 纠正优先于困惑判定（"不对，应该是X"是纠正不是单纯困惑），但两者可共存——
    # 同时命中时都记，delta 累加后由调用方 clamp
    if _hit_any(user, _CORRECTION_PATTERNS):
        signals.append("correction")
    if _hit_any(user, _NEGATIVE_ACK_PATTERNS):
        signals.append("negative_ack")
    if _hit_any(user, _PREFERENCE_PATTERNS):
        signals.append("preference")
    if _hit_any(user, _AFFIRM_PATTERNS):
        signals.append("affirm")

    repeat_count = int((history_hint or {}).get("topic_repeat_count") or 0)
    if repeat_count >= 1:
        signals.append("repeat")

    risk_delta = sum(_RISK_DELTA.get(s, 0.0) for s in signals)
    mastery_delta = sum(_MASTERY_DELTA.get(s, 0.0) for s in signals)
    return {
        "signals": signals,
        "risk_delta": risk_delta,
        "mastery_delta": mastery_delta,
        "topic_repeat_count": repeat_count,
    }


def should_route_error_llm(signal: dict) -> bool:
    """纯函数：该轮是否值得调一次 LLM 抽错题（SAGE 的不确定带）。

    只有强行为证据（教师纠正 / 明确困惑 / 负向反馈 / 重复提问）才路由；偏好线索与
    肯定确认不路由。零 LLM 闭集匹配覆盖其余全部轮次。
    """
    strong = {"correction", "confusion", "negative_ack", "repeat"}
    return bool(strong & set(signal.get("signals") or []))


async def match_topics(
    text: str, course_id: str, db, *, threshold: float | None = None
) -> list[tuple[str, float]]:
    """文本 → 课程主题命中（embedding 最近邻，零 LLM、闭集）。

    每个课程主题的 embedding 已由建库期算好（course_topic.embedding，JSON），这里只算
    一次 query embedding 做余弦最近邻，>= threshold 的按 sim 降序返回 [(topic_id, sim)]。
    embed_model 不可用或课程无主题时返回 []（调用方按"无命中"处理，不阻塞巩固）。
    """
    from core.db.database import CourseTopic
    from core.memory.course_topic_store import TOPIC_MATCH_THRESHOLD, cosine
    from sqlalchemy import select

    if not (text or "").strip():
        return []
    rows = (
        (await db.execute(select(CourseTopic).where(CourseTopic.course_id == course_id)))
        .scalars()
        .all()
    )
    candidates = [(r.topic_id, r.embedding) for r in rows if r.embedding]
    if not candidates:
        return []

    try:
        from core.rag.llamaindex.pg_store import get_embed_model

        q_emb = await get_embed_model()._aget_query_embedding(text)
    except Exception as exc:
        logger.warning("[topic_match] embedding 不可用（返回空命中）：%s", exc)
        return []

    thr = threshold if threshold is not None else TOPIC_MATCH_THRESHOLD
    hits = [(tid, cosine(q_emb, emb)) for tid, emb in candidates]
    hits = [(tid, sim) for tid, sim in hits if sim >= thr]
    hits.sort(key=lambda x: -x[1])
    return hits


def extract_preferences(user_msg: str) -> list[str]:
    """纯函数：抽偏好线索（persona 槽用）。返回命中的原始子句，无则空。

    偏好是跨概念稳定事实（"我喜欢分步讲"），与主题无关，不参与 mastery 门控——
    由 consolidate 写 mem0 persona 槽（阶段2）。
    """
    user = user_msg or ""
    prefs: list[str] = []
    for p in _PREFERENCE_PATTERNS:
        if p in user:
            # 取命中词所在句（中文句号/问号/感叹号/换行切分），去首尾空白
            for sent in re.split(r"[。！？!?\n]", user):
                if p in sent and sent.strip() and sent.strip() not in prefs:
                    prefs.append(sent.strip())
            break  # 一轮最多记一条偏好（多命中同句去重）
    return prefs


__all__ = [
    "match_topics",
    "infer_signal",
    "should_route_error_llm",
    "extract_preferences",
]
