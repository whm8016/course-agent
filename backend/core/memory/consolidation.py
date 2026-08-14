"""L3 巩固：消费 episodic pending → 升格 semantic(mem0) + 掌握度观测 → 标 done。

事件驱动（Phase 3）：热路径 importance 累计超阈值 → enqueue consolidate_memory；
5min cron safety net 兜底长期 pending + 超时 processing 孤儿（见 worker.cron_consolidate_memory）。

掌握度路径（plan「掌握度估计器重构」阶段C）：按主题连续段一次 LLM 闭集序数判定
（judge.judge_segment，一段一次调用）→ 写 mastery_observation 追加日志 → 确定性
Elo 折叠进 knowledge_mastery（mastery_state.fold_observations）。infer_signal 词表
已退出掌握度主路径（降级为待实测 LF，见 topic_match.py docstring）。

预算约束：单次巩固判定段数上限 _JUDGE_SEGMENTS_PER_RUN（超出的段留 pending 等下轮，
掌握度晚一轮到账无害）+ 信号量并发 + 单段超时；判定失败不影响 mem0 与 done 标记。

幂等性：mem0.add 内部做事实抽取/去重，重复巩固同一段不产生重复事实；故崩溃重试安全。
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# 单次巩固最多领取的 episode 数（防爆）
_CLAIM_LIMIT = 200

# 判定预算（plan 阶段C）：200 条 episode 切完可能是几十段，串行几十次 LLM 调用会
# 拖爆 ARQ 任务超时。超上限的段本次不判定，episode 留 pending 等下轮（宁可多轮
# 巩固，不要单任务超时全丢）。
_JUDGE_SEGMENTS_PER_RUN = 20
_JUDGE_CONCURRENCY = 4    # 并发判定信号量
_JUDGE_TIMEOUT_SECONDS = 30  # 单段超时；超时段按判定不可用处理
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

    已不在掌握度主路径上（阶段C 改走 judge → mastery_observation），保留供 eval/
    看板合成路径复用。教师看板读 users.knowledge_graph，节点闭集合成——不为看板
    另跑一遍开放抽取（开放抽取的 entity_id 不稳定，正是 kp_id 断链的成因）。
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


async def _write_mastery_observations(
    db, user_id: str, course_id: str, judgements, episode_objs: list,
    *, source_override: str = "llm_judge", weight_override: float | None = None,
    created_at_override: float | None = None, question_id: str | None = None,
) -> None:
    """观测追加 + 按 (topic) 折叠写回 knowledge_mastery。

    默认走 LLM 判定口径（source=llm_judge，outcome 由等级映射）；quiz_observe 等
    其他来源用 source_override/weight_override 覆写（quiz 的 outcome 是 0/1，不
    经等级映射——由调用方在 judgements 里给 good/poor 对应 1/0 再经 LEVEL_OUTCOME
    还原，语义等价且免双路径）。

    折叠纪律（plan 阶段A）：增量更新只在「新观测就是最新一条」时等价；这里写完即折，
    若有迟到观测落库须整体重折（fold_observations 吃全量日志的设计即为此）。
    失败非致命——观测日志 append-only，丢了可由重放恢复，不能阻塞 done。
    """
    import time as _time

    from core.db.database import KnowledgeMastery, MasteryObservation
    from core.memory.mastery_state import LEVEL_OUTCOME, W_JUDGE
    from sqlalchemy import select

    if not judgements:
        return
    weight = weight_override if weight_override is not None else W_JUDGE
    # 观测时间取段内最后一条 episode 的 created_at（对话发生时刻）而非写入时刻：
    # 巩固延迟批处理时，读侧 p_forget 按「距上次学习」算才对（凌晨批处理若用写入
    # 时间会把 30 天前的对话判成刚学过，p_forget≈0，预警全灭）。
    ep_time = created_at_override if created_at_override is not None else (
        max((getattr(ep, "created_at", 0) or _time.time()) for ep in episode_objs)
        if episode_objs else _time.time()
    )
    for j in judgements:
        db.add(
            MasteryObservation(
                user_id=user_id, course_id=course_id, topic_id=j.topic_id,
                source=source_override, outcome=LEVEL_OUTCOME[j.level], weight=weight,
                confidence=j.confidence, evidence_quote=j.evidence_quote,
                episode_ids=[e.id for e in episode_objs] or None,
                question_id=question_id,
                created_at=ep_time,
            )
        )
    await db.commit()

    # 折叠写回：按 topic 全量重放观测日志 → 更新 theta/n_obs 与派生 mastery/risk
    topic_ids = sorted({j.topic_id for j in judgements})
    from core.memory.mastery_state import fold_observations

    for tid in topic_ids:
        obs_rows = (
            (
                await db.execute(
                    select(
                        MasteryObservation.outcome,
                        MasteryObservation.weight,
                        MasteryObservation.source,
                        MasteryObservation.created_at,
                    ).where(
                        MasteryObservation.user_id == user_id,
                        MasteryObservation.course_id == course_id,
                        MasteryObservation.topic_id == tid,
                    )
                )
            )
            .mappings()
            .all()
        )
        # 题目难度侧（全班共享）：全班（course+topic）quiz 观测从零整体重折（幂等）。
        # 不从 course_topic.difficulty_b 取起点叠加——b 是全班参数，任何 (user, topic)
        # 侧的叠加都会把同一份历史重复折进（起点已是历史折出值，再重放一遍 = 双计）；
        # 也不能只看当前 user 的 quiz——b 要吃全班的判分证据（新生 theta=0 时
        # risk = sigmoid(b) 即班级先验，MasteryObservation 表 docstring 同口径）。
        from core.db.database import CourseTopic

        class_quiz = (
            (
                await db.execute(
                    select(
                        MasteryObservation.outcome,
                        MasteryObservation.weight,
                        MasteryObservation.source,
                        MasteryObservation.created_at,
                    ).where(
                        MasteryObservation.course_id == course_id,
                        MasteryObservation.topic_id == tid,
                        MasteryObservation.source == "quiz",
                    )
                )
            )
            .mappings()
            .all()
        )
        ct = (
            (
                await db.execute(
                    select(CourseTopic.id).where(
                        CourseTopic.course_id == course_id,
                        CourseTopic.topic_id == tid,
                    )
                )
            ).first()
        )
        item_state = fold_observations(class_quiz)  # b 从 0 起、全班 quiz 全量、可重放
        all_obs = [
            {"created_at": o.created_at, "outcome": o.outcome, "weight": o.weight, "source": o.source}
            for o in obs_rows
        ]
        # 学生侧：难度冻结为 0 只折 theta——行上 risk/mastery 由读侧用 course_topic
        # 的全班 b 派生（proactive._read_mastery 同口径），单一真相源不在行上另存
        state = fold_observations(all_obs, freeze_item=True)
        # 题目侧（全班共享）：写回 course_topic（单一真相源）
        if item_state.n_item > 0 and ct is not None:
            from sqlalchemy import update

            await db.execute(
                update(CourseTopic)
                .where(
                    CourseTopic.course_id == course_id,
                    CourseTopic.topic_id == tid,
                )
                .values(
                    difficulty_b=round(item_state.b_topic, 6),
                    n_item=item_state.n_item,
                )
            )

        row = (
            (
                await db.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == user_id,
                        KnowledgeMastery.course_id == course_id,
                        KnowledgeMastery.kp_id == tid,
                    )
                )
            )
            .scalars()
            .first()
        )
        if row is None:
            from core.db.database import CourseTopic as _CT

            label_row = (
                await db.execute(
                    select(_CT.label).where(
                        _CT.course_id == course_id, _CT.topic_id == tid
                    )
                )
            ).scalar()
            db.add(
                KnowledgeMastery(
                    user_id=user_id, course_id=course_id, kp_id=tid,
                    label=label_row or tid,
                    mastery=round(state.mastery, 4), risk=round(state.risk, 4),
                    observation_count=state.n_obs,
                    first_observed_at=ep_time, last_observed_at=ep_time,
                    evidence_episode_ids=[e.id for e in episode_objs][:20],
                    theta=round(state.theta, 6), n_obs=state.n_obs,
                )
            )
        else:
            row.theta = round(state.theta, 6)
            row.n_obs = state.n_obs
            row.mastery = round(state.mastery, 4)
            row.risk = round(state.risk, 4)
            row.observation_count = state.n_obs
            # last_observed_at 单调不回退：迟到观测重折不把时间戳拉回过去
            if ep_time > (row.last_observed_at or 0):
                row.last_observed_at = ep_time
            ev = list(row.evidence_episode_ids or [])
            for e in episode_objs:
                if e.id and e.id not in ev:
                    ev.append(e.id)
            row.evidence_episode_ids = ev[-20:]
        await db.commit()


async def _judge_and_record(
    db, user_id: str, course_id: str, segments: list, topic_ids_per_ep: list,
    topics_meta: dict[str, tuple[str, str]], *,
    _hits_map: dict[str, list], _eps_all: list, budget_left: list[int],
) -> list[str]:
    """按主题段判定 + 写观测 + 折叠。返回判定不可用（超时/失败/超上限/写库失败）段的
    episode id 列表——这些 episode 留 pending 等下轮。

    topics_meta: {topic_id: (label, definition)}，judge 的闭集输入。
    _hits_map: {episode_id: [topic_id, ...]}，段内可判主题 = 段内各 ep 命中并集（≤3）。
    _eps_all: 该 session 全部 episode（topic_ids_per_ep 的索引基准）。
    budget_left: 长度 1 的箱子，装「本次 consolidate 还可判定的段数」——预算在
    **整次巩固**内共享（跨 session），防 200 条 episode × N session 放大到 N×20 次
    LLM 拖爆 ARQ。函数内原地扣减。

    并发模型：LLM 判定（纯 IO）用信号量并发；DB 写回**串行**（write_lock）——SQLAlchemy
    AsyncSession 非并发安全，gather 共享同一 session 会撞事务状态机。写回本身轻
    （append + 全量重折在观测数小时级），串行不构成瓶颈。
    """
    from core.memory.judge import judge_segment

    n_budget = min(len(segments), max(0, budget_left[0]))
    budget_left[0] -= n_budget
    overflow = [ep.id for seg in segments[n_budget:] for ep in seg]
    if overflow:
        logger.info(
            "[consolidate] judge budget overflow: %d segments deferred",
            len(segments) - n_budget,
        )
    budget = segments[:n_budget]

    sem = asyncio.Semaphore(_JUDGE_CONCURRENCY)
    write_lock = asyncio.Lock()
    deferred_ids: list[str] = list(overflow)
    first_idx = {id(ep): i for i, ep in enumerate(_eps_all)}

    async def _one(seg) -> None:
        # 段内 episode id 先快照：rollback 会 expire ORM 实例，之后再读 ep.id 触发
        # 同步刷新 IO，在 async 上下文抛 MissingGreenlet（连带 guard/外层连环崩）。
        seg_ids = [ep.id for ep in seg]
        async with sem:
            main_topic = topic_ids_per_ep[first_idx[id(seg[0])]]
            if main_topic is None:
                return  # 无主题段无可判主题
            seg_topics = sorted(
                {t for ep in seg for t in (_hits_map.get(ep.id) or [])}
            )[:3] or [main_topic]
            if not all(t in topics_meta for t in seg_topics):
                return
            try:
                result = await asyncio.wait_for(
                    judge_segment(
                        [(ep.user_msg, ep.assistant_msg) for ep in seg],
                        [(t, *topics_meta[t]) for t in seg_topics],
                    ),
                    timeout=_JUDGE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning("[consolidate] judge timeout, segment deferred")
                deferred_ids.extend(seg_ids)
                return
            except Exception as exc:
                logger.warning("[consolidate] judge failed: %s", exc)
                deferred_ids.extend(seg_ids)
                return
            if result is None:
                deferred_ids.extend(seg_ids)
                return
            async with write_lock:
                try:
                    await _write_mastery_observations(
                        db, user_id, course_id, result.judgements, list(seg)
                    )
                except Exception as exc:
                    # 写库失败必须延期：观测没落库就标 done，这段对话的掌握度证据
                    # 永久丢失（episode 不再被巩固领走）。回 pending 等下轮重判重写。
                    logger.warning("[consolidate] mastery write failed: %s", exc)
                    await db.rollback()
                    deferred_ids.extend(seg_ids)
                    return
                if result.error_patterns:
                    try:
                        from core.memory import graph_memory

                        await graph_memory.merge_and_save_graphs(
                            db, user_id,
                            {"error_patterns": result.error_patterns},
                            course_id,
                        )
                    except Exception as exc:
                        logger.warning("[consolidate] error_patterns merge failed: %s", exc)
                        await db.rollback()

    # 段级 try 兜底：单段意外异常不拖垮整批（其余段照常判定/写回）
    async def _guard(seg) -> None:
        seg_ids = [ep.id for ep in seg]  # 同上：rollback/expire 后 ep.id 不可再读
        try:
            await _one(seg)
        except Exception as exc:
            logger.warning("[consolidate] segment judge crashed: %s", exc)
            deferred_ids.extend(seg_ids)

    await asyncio.gather(*(_guard(seg) for seg in budget))
    return sorted(set(deferred_ids))


async def consolidate(db, user_id: str, course_id: str = "") -> dict[str, int]:
    """核心巩固流程（ARQ job 调用，便于单测）。

    claim pending → 按 session 分组 → mem0 升格 + 按主题段 LLM 判定写观测折叠
    → 成功标 done / 判定不可用的 episode 回 pending 等下轮。

    Returns: {"claimed": N, "promoted": M}
    """
    claimed = await claim_pending(db, user_id, course_id)
    if not claimed:
        return {"claimed": 0, "promoted": 0}

    done_ids: list[str] = []
    deferred_ids: set[str] = set()
    promoted = 0
    # 判定段预算：整次 consolidate 共享 _JUDGE_SEGMENTS_PER_RUN（非 per-session）——
    # 200 条 episode × 多 session 若每 session 各判 20 段，LLM 次数随 session 数
    # 线性放大，正是该预算要防的 ARQ 任务超时。超预算的段留 pending 等下轮。
    judge_budget = [_JUDGE_SEGMENTS_PER_RUN]

    from core.memory.topic_match import match_topics

    for session_id, eps in group_episodes_by_session(claimed).items():
        # id 先快照：rollback 会 expire ORM 实例，之后再读 e.id 触发同步刷新 IO 抛
        # MissingGreenlet（async 上下文不能同步 lazy load）
        eps_ids = [e.id for e in eps]
        # 0. 逐 episode 闭集命中（每轮 match_topics 只算一次，判定与双槽写入共用）
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
        # 2. 掌握度路径（一段一次 LLM 判定 → 观测 → 折叠）。失败非致命，不影响 done
        try:
            segments = split_topic_segments(eps, topic_ids_per_ep)
            # 段内可判主题 = 段内各 ep 命中主题的并集（闭集 meta 一次查齐）
            seg_topic_ids = sorted(
                {
                    t
                    for hits in hits_per_ep
                    for t, _sim in hits
                }
            )
            topics_meta: dict[str, tuple[str, str]] = {}
            if seg_topic_ids:
                from core.db.database import CourseTopic
                from sqlalchemy import select

                rows = (
                    (
                        await db.execute(
                            select(
                                CourseTopic.topic_id, CourseTopic.label, CourseTopic.definition
                            ).where(
                                CourseTopic.course_id == course_id,
                                CourseTopic.topic_id.in_(seg_topic_ids),
                            )
                        )
                    ).all()
                )
                topics_meta = {
                    r.topic_id: (r.label, r.definition or "") for r in rows
                }
            hits_by_ep = {ep.id: [t for t, _s in hits] for ep, hits in zip(eps, hits_per_ep)}
            if topics_meta:
                deferred = await _judge_and_record(
                    db, user_id, course_id, segments, topic_ids_per_ep, topics_meta,
                    _hits_map=hits_by_ep, _eps_all=eps, budget_left=judge_budget,
                )
                deferred_ids.update(deferred)
        except Exception as exc:
            # 整段 judge 路径抛错（切段/查 meta）→ 该 session 全部 episode 回 pending：
            # 标 done 会让这段对话的掌握度观测永久丢失（episode 不再被巩固领走）。
            # mem0 可能已写（幂等，重放不产生重复事实），掌握度下轮补上即可。
            logger.warning(
                "[consolidate] judge path failed user=%s session=%s error=%s",
                user_id, session_id, exc,
            )
            deferred_ids.update(eps_ids)
            try:
                await db.rollback()
            except Exception:
                pass
        done_ids.extend(eps_ids)
        promoted += len(eps)

    # 判定不可用的 episode 回 pending（下轮重判）；其余 done
    done_final = [i for i in done_ids if i not in deferred_ids]
    await _mark(db, done_final, "done", consolidated=True)
    await _mark(db, sorted(deferred_ids), "pending")

    logger.info(
        "[consolidate] user=%s course=%s claimed=%d promoted=%d deferred=%d",
        user_id, course_id, len(claimed), len(done_final), len(deferred_ids),
    )
    return {"claimed": len(claimed), "promoted": len(done_final)}

    # procedural 草稿生成已从主路径摘除：学情拼接门控方案下 procedural 不再作为贡献，
    # 故 consolidate 不再调 maybe_generate_procedural（模块 procedural.py 保留，避免大删
    # 前端技能体系）。如需恢复，在此处重新 ``await maybe_generate_procedural(db, user_id, course_id)``。


__all__ = [
    "group_episodes_by_session",
    "add_importance",
    "reset_importance",
    "claim_pending",
    "consolidate",
    "_PENDING_STALE_SECONDS",
    "_PROCESSING_TIMEOUT_SECONDS",
]
