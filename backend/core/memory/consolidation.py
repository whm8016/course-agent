"""L3 巩固：消费 episodic pending → 升格 semantic(mem0) → 标 done。

事件驱动（Phase 3）：热路径 importance 累计超阈值 → enqueue consolidate_memory；
5min cron safety net 兜底长期 pending + 超时 processing 孤儿（见 worker.cron_consolidate_memory）。

写侧闭集化（零 LLM）：主题命中走 topic_match.match_topics（课程主题图 embedding 最近邻），
行为信号走 infer_signal 关键词规则；仅 should_route_error_llm 判强错误信号时才调一次 LLM
抽错题（且只取 error_patterns）。抽象已由建库期课程主题图提供，运行时不再生成式抽取。

幂等性：mem0.add 内部做事实抽取/去重，重复巩固同一段不产生重复事实；故崩溃重试安全。
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# 单次巩固最多领取的 episode 数（防爆）
_CLAIM_LIMIT = 200
# pending 超过此秒数 → cron safety net 兜底 enqueue（session 空闲等价触发）
_PENDING_STALE_SECONDS = 300
# processing 超过此秒数（用 created_at 近似，正常 promote 仅需数秒）→ 视崩溃遗留回 pending
_PROCESSING_TIMEOUT_SECONDS = 600

# importance 累计 Redis key（事件驱动触发用）
_IMPORTANCE_KEY = "mem_importance:"
_IMPORTANCE_TTL = 86400


def group_episodes_by_session(episodes: list) -> dict[str, list]:
    """按 session_id 分组（空 session 归入 _no_session）。组内保持调用方传入的顺序。"""
    groups: dict[str, list] = defaultdict(list)
    for ep in episodes:
        groups[ep.session_id or "_no_session"].append(ep)
    return dict(groups)


async def add_importance(user_id: str, course_id: str, delta: float) -> float:
    """Redis INCRBYFLOAT 累计该 user(+course) 的待巩固 importance，返回累计值。"""
    from core.memory.redis_pool import get_redis as _get_redis

    r = _get_redis()
    key = f"{_IMPORTANCE_KEY}{user_id}:{course_id}"
    pipe = r.pipeline()
    pipe.incrbyfloat(key, delta)
    pipe.expire(key, _IMPORTANCE_TTL)
    new, _ = await pipe.execute()
    return float(new)


async def reset_importance(user_id: str, course_id: str) -> None:
    """巩固触发后清零累计（下一轮重新攒）。"""
    from core.memory.redis_pool import get_redis as _get_redis

    r = _get_redis()
    await r.delete(f"{_IMPORTANCE_KEY}{user_id}:{course_id}")


async def claim_pending(db, user_id: str, course_id: str, *, limit: int = _CLAIM_LIMIT) -> list:
    """原子领取 pending episodes → 标 processing（条件 UPDATE + RETURNING 实际命中行）。

    SELECT 候选 → UPDATE ... WHERE id IN 候选 AND status='pending' RETURNING id。
    返回**真正被本次 UPDATE 命中的行**（RETURNING 结果），而非 SELECT 快照——治并发
    重复巩固：两 worker 同 SELECT 同一批候选时，先 commit 者把 status 翻成 processing，
    后者 UPDATE 的 status='pending' 条件不再命中（Postgres 取行锁后重查 WHERE），
    RETURNING 空 → 后者返回 []，不会重复喂 mem0（双倍 LLM 成本 + 重复事实）。
    """
    from core.db.database import MemoryEpisode
    from sqlalchemy import select, update

    rows = (
        (
            await db.execute(
                select(MemoryEpisode)
                .where(
                    MemoryEpisode.user_id == user_id,
                    MemoryEpisode.status == "pending",
                    *([MemoryEpisode.course_id == course_id] if course_id else []),
                )
                .order_by(MemoryEpisode.created_at)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return []
    result = await db.execute(
        update(MemoryEpisode)
        .where(
            MemoryEpisode.id.in_([r.id for r in rows]),
            MemoryEpisode.status == "pending",
        )
        .values(status="processing")
        .returning(MemoryEpisode.id)
        .execution_options(synchronize_session=False)
    )
    claimed_ids = set(result.scalars().all())
    await db.commit()
    # 只返回真正被本次 UPDATE 命中的行（候选中被并发抢走的剔除），治并发重复巩固
    return [r for r in rows if r.id in claimed_ids]


def split_topic_segments(episodes: list, topic_ids_per_ep: list) -> list[list]:
    """按主题连续段切分（纯函数，SeCom 的 segment 粒度，边界=课程主题变更）。

    Args:
        episodes: 同 session 按序的 episode 列表。
        topic_ids_per_ep: 每条 episode 的主 topic（match_topics 最高 sim 的 id，无命中 None）。

    规则：主 topic 与上一条不同就切开；None 自成一段（通常不写 event）。
    返回段列表，每段是连续的 episode 子列表。空输入返回 []。
    """
    if len(episodes) != len(topic_ids_per_ep):
        raise ValueError("episodes 与 topic_ids_per_ep 长度不一致")
    segments: list[list] = []
    cur: list = []
    cur_topic = object()  # 哨兵：与任何 topic（含 None）都不等，首条必开新段
    for ep, main_topic in zip(episodes, topic_ids_per_ep):
        if main_topic != cur_topic:
            if cur:
                segments.append(cur)
            cur, cur_topic = [], main_topic
        cur.append(ep)
    if cur:
        segments.append(cur)
    return segments


def _event_line(seg: list, main_topic: str, label: str, signals: list) -> str:
    """一段压成一行结构化文本（规则压缩，不接 LLM 摘要——SeCom 自己说 summary 丢细节）。"""
    tags = "/".join(dict.fromkeys(s for s in signals if s)) or "学习"
    return f"{label} · {len(seg)}轮 · {tags}"


async def _promote_segment(
    user_id: str, course_id: str, episodes: list, db=None, topic_ids_per_ep: list | None = None
) -> None:
    """episodes → mem0 双槽写入（persona/event 分存，infer=False 零 LLM）。

    - kind=event：一个主题连续段一条（split_topic_segments 切），文本规则压一行
    - kind=persona：偏好线索一条事实一条（extract_preferences），不按段切
    topic_ids_per_ep 由 consolidate 算好传入（match_topics 每轮只跑一次，不双倍 embedding）；
    未传且无 db 时按无命中处理（eval/单测路径）。回填段内 episode 的 segment_id/topic_ids。
    """
    import uuid

    from core.memory.topic_match import extract_preferences, infer_signal, match_topics

    if topic_ids_per_ep is None:
        topic_ids_per_ep = []
        if db is not None:
            for ep in episodes:
                hits = await match_topics(
                    f"{ep.user_msg}\n{ep.assistant_msg}", course_id, db
                )
                topic_ids_per_ep.append(hits[0][0] if hits else None)
        else:
            topic_ids_per_ep = [None] * len(episodes)

    # label 查询一次批拉（段内主题 → 中文名）
    label_by_topic: dict[str, str] = {}
    topics_with_label = sorted({t for t in topic_ids_per_ep if t})
    if db is not None and topics_with_label:
        from core.db.database import CourseTopic
        from sqlalchemy import select

        rows = (
            await db.execute(
                select(CourseTopic.topic_id, CourseTopic.label).where(
                    CourseTopic.course_id == course_id,
                    CourseTopic.topic_id.in_(topics_with_label),
                )
            )
        ).all()
        label_by_topic = dict(rows)

    from core.memory.mem0_client import get_memory

    m = get_memory()
    persona_done: set[str] = set()  # 同一偏好句只写一次（跨段去重）
    for seg, main_topic in iter_segments_with_topic(episodes, topic_ids_per_ep):
        seg_id = uuid.uuid4().hex[:16]
        signals: list[str] = []
        for ep in seg:
            sig = infer_signal(ep.user_msg, ep.assistant_msg)
            signals.extend(sig["signals"])
            for pref in extract_preferences(ep.user_msg):
                if pref not in persona_done:
                    persona_done.add(pref)
                    await m.add(
                        pref,
                        user_id=user_id,
                        metadata={
                            "course_id": course_id, "kind": "persona", "source": "consolidate",
                        },
                        infer=False,
                    )
        if main_topic is None:
            continue  # 无主题段不写 event（事务性寒暄/跑题）
        label = label_by_topic.get(main_topic) or main_topic
        await m.add(
            _event_line(seg, main_topic, label, signals),
            user_id=user_id,
            metadata={
                "course_id": course_id, "kind": "event", "source": "consolidate",
                "topic_ids": [main_topic], "segment_id": seg_id,
            },
            infer=False,
        )
        # 回填段内 episode 的 segment_id 与 topic_ids
        if db is not None:
            from core.db.database import MemoryEpisode
            from sqlalchemy import update

            await db.execute(
                update(MemoryEpisode)
                .where(MemoryEpisode.id.in_([e.id for e in seg]))
                .values(segment_id=seg_id, topic_ids=[main_topic])
            )
            await db.commit()


def iter_segments_with_topic(episodes: list, topic_ids_per_ep: list):
    """yield (段, 该段主 topic)——段与其主题成对迭代，免去调用方按下标回查。"""
    for seg in split_topic_segments(episodes, topic_ids_per_ep):
        # 段内主题一致（切分规则保证），取首条的主题即段主题
        first_idx = next(i for i, e in enumerate(episodes) if e is seg[0])
        yield seg, topic_ids_per_ep[first_idx]


async def _mark(db, ep_ids: list[str], status: str, *, consolidated: bool = False) -> None:
    if not ep_ids:
        return
    from core.db.database import MemoryEpisode
    from sqlalchemy import update

    values: dict[str, Any] = {"status": status}
    if consolidated:
        values["consolidated_at"] = time.time()
    await db.execute(update(MemoryEpisode).where(MemoryEpisode.id.in_(ep_ids)).values(**values))
    await db.commit()


async def _closed_set_points(db, course_id: str, hits: list, signal: dict) -> list[dict]:
    """闭集命中 → 仪表盘 knowledge 节点（entity_id=topic_id，id 永不漂移）。

    教师看板读 users.knowledge_graph，节点改为闭集合成——不为看板另跑一遍开放抽取
    （开放抽取每次抽出的 entity_id 不稳定，正是 kp_id 断链的成因）。label 查不到的跳过。
    """
    from core.db.database import CourseTopic
    from sqlalchemy import select

    tids = [tid for tid, _sim in hits]
    if not tids:
        return []
    label_map = dict(
        (
            await db.execute(
                select(CourseTopic.topic_id, CourseTopic.label).where(
                    CourseTopic.course_id == course_id,
                    CourseTopic.topic_id.in_(tids),
                )
            )
        ).all()
    )
    return [
        {
            "entity_id": tid,
            "label": label_map[tid],
            "mastery_delta": signal.get("mastery_delta", 0.0),
            "risk_delta": signal.get("risk_delta", 0.0),
        }
        for tid in tids
        if label_map.get(tid)
    ]


async def consolidate(db, user_id: str, course_id: str = "") -> dict[str, int]:
    """核心巩固流程（ARQ job 调用，便于单测）。

    claim pending → 按 session 分组 → 每组 mem0.add 升格 → 成功标 done / 失败回 pending 重试。
    mem0 失败不丢数据：失败的 episode 回 pending 等下次重试（episodes 永久留存）。

    Returns: {"claimed": N, "promoted": M}
    """
    claimed = await claim_pending(db, user_id, course_id)
    if not claimed:
        return {"claimed": 0, "promoted": 0}

    done_ids: list[str] = []
    retry_ids: list[str] = []
    promoted = 0
    from core.memory import graph_memory, mastery  # 派生层：graph(dashboard) + mastery(course 维度)

    from core.memory.topic_match import (
        infer_signal,
        match_topics,
        should_route_error_llm,
    )

    for session_id, eps in group_episodes_by_session(claimed).items():
        # 0. 逐 episode 闭集命中（每轮 match_topics 只算一次，双槽写入与 mastery 共用）
        hits_per_ep: list = []
        for ep in eps:
            try:
                hits_per_ep.append(
                    await match_topics(
                        f"{ep.user_msg}\n{ep.assistant_msg}", course_id, db
                    )
                )
            except Exception as exc:
                logger.warning("[consolidate] match_topics failed ep=%s: %s", ep.id, exc)
                hits_per_ep.append([])
        topic_ids_per_ep = [hits[0][0] if hits else None for hits in hits_per_ep]

        # 1. mem0 双槽升格 —— 失败只降级 warning：不再整段回 pending 拖垮 mastery
        #    （旧实现 mem0 失败会连带丢掉本可写入的掌握度观测；episodes 永久留存，
        #    mem0 段丢失可由 episodic 层重放，掌握度观测丢了就是丢了）
        try:
            await _promote_segment(user_id, course_id, eps, db, topic_ids_per_ep)
        except Exception as exc:
            logger.warning(
                "[consolidate] mem0 promote failed user=%s session=%s error=%s",
                user_id, session_id, exc,
            )
        # 2. graph + mastery（派生层）。写侧闭集化（零 LLM）：match_topics 定主题，
        #    infer_signal 出行为 delta；仅强错误信号（should_route_error_llm）才调 LLM
        #    抽错题（且只取 error_patterns，丢弃 knowledge_points——开放抽取的 entity_id
        #    不稳定，正是 kp_id 断链的成因）。失败非致命，不影响 done。
        for ep, hits in zip(eps, hits_per_ep):
            try:
                # 重复提问提示：命中的主题此前已有 mastery 观测（本轮写入前）
                repeat = await mastery.topic_repeat_hint(
                    db, user_id, course_id, [tid for tid, _sim in hits]
                )
                signal = infer_signal(
                    ep.user_msg, ep.assistant_msg,
                    history_hint={"topic_repeat_count": repeat},
                )
                if hits:
                    await mastery.append_mastery_from_topics(
                        db, user_id, course_id, hits, signal, ep.id
                    )
                    points = await _closed_set_points(db, course_id, hits, signal)
                    if points:
                        await graph_memory.merge_and_save_graphs(
                            db, user_id, {"knowledge_points": points}, course_id
                        )
                if should_route_error_llm(signal):
                    extracted = await graph_memory.extract_knowledge(
                        course_id, ep.user_msg, ep.assistant_msg
                    )
                    err_only = {"error_patterns": (extracted or {}).get("error_patterns") or []}
                    if err_only["error_patterns"]:
                        await graph_memory.merge_and_save_graphs(db, user_id, err_only, course_id)
            except Exception as exc:
                logger.warning(
                    "[consolidate] knowledge update failed user=%s ep=%s: %s",
                    user_id, ep.id, exc,
                )
                try:
                    await db.rollback()  # 失败可能令 session 中毒，回滚后续 episode/_mark 才能用
                except Exception:
                    pass
        done_ids.extend(e.id for e in eps)
        promoted += len(eps)

    await _mark(db, done_ids, "done", consolidated=True)
    await _mark(db, retry_ids, "pending")  # 回退重试

    # procedural 草稿生成已从主路径摘除：学情拼接门控方案下 procedural 不再作为贡献，
    # 故 consolidate 不再调 maybe_generate_procedural（模块 procedural.py 保留，避免大删
    # 前端技能体系）。如需恢复，在此处重新 ``await maybe_generate_procedural(db, user_id, course_id)``。

    logger.info(
        "[consolidate] user=%s course=%s claimed=%d promoted=%d retry=%d",
        user_id, course_id, len(claimed), promoted, len(retry_ids),
    )
    return {"claimed": len(claimed), "promoted": promoted}


__all__ = [
    "group_episodes_by_session",
    "add_importance",
    "reset_importance",
    "claim_pending",
    "consolidate",
    "_PENDING_STALE_SECONDS",
    "_PROCESSING_TIMEOUT_SECONDS",
]
