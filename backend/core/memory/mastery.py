"""L3 mastery 层：知识点掌握度（course 维度），追加观测 + 读时指数衰减。

append_mastery：从抽取的知识点追加观测（不覆盖历史——blend mastery/risk + observation_count
累加 + evidence_episode_ids 追加）。course 隔离修 users.knowledge_graph 的跨课程污染。

get_mastery_context：读时按 last_observed_at 做指数衰减（effective_risk = risk · exp(-λ·age)），
取 risk 最高的 top-N 薄弱点拼成 ≤300 字注入 prompt（Phase 4b 用）。软衰减只影响排序，
不物理删除——三个月前的错误概念正是诊断反复性错误的关键证据（Graphiti bi-temporal）。
"""
from __future__ import annotations

import hashlib
import logging
import math
import time

logger = logging.getLogger(__name__)

# 读时衰减系数（按天）：0.01 → 半衰期约 69 天。旧错误概念软降权，不物理删除。
# 注意：λ 只用于 get_mastery_context 的**排序**；门控的遗忘项走 forget_prob
#（MemoryBank 口径，衰减的是学生掌握而非 agent 记忆），两者勿混。
_DECAY_LAMBDA_PER_DAY = 0.01

# MemoryBank 遗忘强度初值（天）：R = exp(-t/S)，teacher_affirm 时 +1（spacing effect）
DEFAULT_STRENGTH = 5.0


def forget_prob(age_days: float, strength: float = DEFAULT_STRENGTH) -> float:
    """遗忘概率 p_forget = 1 - R = 1 - exp(-t/S)（纯函数）。

    t 越久越可能忘了（时间越久越**该**拼前置提醒——方向与旧 eff_risk=risk·exp(-λt)
    相反：旧式衰减把老薄弱点排到后面，恰好埋掉最需要预警的点）。S 越大（被肯定/
    强化越多）忘得越慢。
    """
    t = max(0.0, float(age_days or 0.0))
    s = max(0.1, float(strength or DEFAULT_STRENGTH))
    return 1.0 - math.exp(-t / s)


def _clamp(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _delta(v) -> float:
    """delta 原值（可负），仅 float 转换不钳制——负 delta 表示退步/改善，钳掉会丢信号。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _kp_id_from(kp: dict) -> str:
    entity_id = str(kp.get("entity_id") or "").strip()
    if entity_id:
        return entity_id
    label = str(kp.get("label") or "").strip()
    return "kp:" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]


async def append_mastery(
    db, user_id: str, course_id: str, knowledge_points: list, episode_id: str
) -> int:
    """[DEPRECATED] 旧 hash 口径的掌握度写入。只剩 eval_memory 评测种子在用。

    **勿再接入新调用方**：主路径是 mastery_observation 追加日志 + Elo 折叠
    （consolidation._write_mastery_observations），本函数直接 blend 行值，
    与闭集口径并存会污染 knowledge_mastery。评测侧也应逐步改走观测日志。

    支持两种抽取格式：
    - catalog(delta)：{entity_id, label, mastery_delta, risk_delta} → 累加 delta
    - fallback(绝对值)：{label, mastery, risk} → 与现有值平均
    幂等键 (user_id, course_id, kp_id)：已有行则 blend+count++，否则新建。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import select

    if not knowledge_points:
        return 0
    now = time.time()
    n = 0
    for kp in knowledge_points:
        label = str(kp.get("label") or "").strip()
        if not label:
            continue
        kp_id = _kp_id_from(kp)
        is_delta = "mastery_delta" in kp or "risk_delta" in kp
        m_delta = _delta(kp.get("mastery_delta")) if is_delta else 0.0
        r_delta = _delta(kp.get("risk_delta")) if is_delta else 0.0
        m_abs = _clamp(kp.get("mastery"), 0.5) if not is_delta else 0.5
        r_abs = _clamp(kp.get("risk"), 0.5) if not is_delta else 0.5

        row = (
            (
                await db.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == user_id,
                        KnowledgeMastery.course_id == course_id,
                        KnowledgeMastery.kp_id == kp_id,
                    )
                )
            )
            .scalars()
            .first()
        )

        if row is not None:
            # 追加观测：delta 累加 / 绝对值平均；evidence 追加（不覆盖历史）
            if is_delta:
                row.mastery = _clamp(row.mastery + m_delta, 0.5)
                row.risk = _clamp(row.risk + r_delta, 0.5)
            else:
                row.mastery = round((row.mastery + m_abs) / 2, 3)
                row.risk = round((row.risk + r_abs) / 2, 3)
            row.observation_count = (row.observation_count or 1) + 1
            row.last_observed_at = now
            ev = list(row.evidence_episode_ids or [])
            if episode_id and episode_id not in ev:
                ev.append(episode_id)
            row.evidence_episode_ids = ev[-20:]
        else:
            mastery_v = _clamp(0.5 + m_delta, 0.5) if is_delta else m_abs
            risk_v = _clamp(0.5 + r_delta, 0.5) if is_delta else r_abs
            db.add(
                KnowledgeMastery(
                    user_id=user_id,
                    course_id=course_id,
                    kp_id=kp_id,
                    label=label,
                    mastery=mastery_v,
                    risk=risk_v,
                    observation_count=1,
                    first_observed_at=now,
                    last_observed_at=now,
                    evidence_episode_ids=[episode_id] if episode_id else [],
                )
            )
        n += 1
    await db.commit()
    return n


async def get_weak_topics(
    db, user_id: str, course_id: str, *, top_n: int = 5
) -> list[dict]:
    """读时衰减后 risk 最高的 top-N 薄弱点（结构化，单一真相源）。

    get_mastery_context（拼文本注入）与出题 plan 阶段（定向出题 + target_topic_id
    对账）共用本函数，避免两处各写一遍衰减+筛选。返回
    [{topic_id, label, risk, observation_count}]，按衰减后 effective_risk 降序。

    薄弱点：衰减后 effective_risk 高优先；risk<0.5 的不算薄弱
    注意：此 0.5 是**仪表盘排序口径**，与门控的 WEAK_MARGIN（logit 空间，plan §3e）
    故意不同源——门控判据是「至少一次判 poor」的语义锚点（2026-08-18 由两次下调），
    这里只是看板筛选线。
    勿"顺手统一"：把门控判据拉回 0.5 会把表现平平（fair）误判成前置缺口。

    topic_id 取行上 kp_id（新口径闭集 id 即 topic_id）。legacy hash 行不做 label
    对齐：出题展示走 label，id 断链只影响旧数据对账，不阻塞主流程（门控读侧的对
    齐在 proactive._read_mastery，口径不同勿搬）。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import select

    rows = (
        (
            await db.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == user_id,
                    KnowledgeMastery.course_id == course_id,
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return []

    now = time.time()
    scored = []
    for r in rows:
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        decay = math.exp(-_DECAY_LAMBDA_PER_DAY * age_days)
        scored.append((r, r.risk * decay))

    scored.sort(key=lambda x: -x[1])
    weak = [(r, eff) for (r, eff) in scored if r.risk >= 0.5][:top_n]
    return [
        {
            "topic_id": r.kp_id,
            "label": r.label,
            "risk": round(float(r.risk or 0.0), 4),
            "observation_count": r.observation_count or 1,
        }
        for r, _eff in weak
    ]


async def get_mastery_context(
    db, user_id: str, course_id: str, *, top_n: int = 5, max_chars: int = 300
) -> str:
    """读时衰减后取 risk 最高的 top-N 薄弱点，拼成注入文本（Phase 4b 用）。无数据则空串。"""
    weak = await get_weak_topics(db, user_id, course_id, top_n=top_n)
    if not weak:
        return ""

    lines = ["## 该生掌握度（薄弱点优先，注意诊断反复性错误）"]
    for w in weak:
        lines.append(
            f"- {w['label']}：偏薄弱(风险{w['risk']:.2f}，{w['observation_count']}次观测)"
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


async def topic_repeat_hint(db, user_id: str, course_id: str, topic_ids: list[str]) -> int:
    """这些主题在该生已有 mastery 行的最大 observation_count（本轮写入前），无行则 0。

    供 infer_signal 的 history_hint——同主题此前观测过 ≥1 次又来问 = 重复提问信号。
    不查 learning_events（那要 join 解析，成本高且口径松）。

    注意：掌握度主路径已改走 judge → mastery_observation（见 consolidation），本函数
    仅剩词表 LF 兜底线（待阶段 E 实测 precision 定去留）使用。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import func, select

    if not topic_ids:
        return 0
    val = (
        await db.execute(
            select(func.max(KnowledgeMastery.observation_count)).where(
                KnowledgeMastery.user_id == user_id,
                KnowledgeMastery.course_id == course_id,
                KnowledgeMastery.kp_id.in_(topic_ids),
            )
        )
    ).scalar()
    return int(val or 0)


async def append_mastery_from_topics(
    db, user_id: str, course_id: str, hits: list[tuple[str, float]], signal: dict, episode_id: str
) -> int:
    """闭集口径写 mastery：kp_id = topic_id（禁止再哈希），label 从 CourseTopic 查。

    .. deprecated:: 死代码——全仓库无生产调用方（词表赋值已随掌握度估计器重构退出主路径，
    巩固写侧统一走 ``consolidation._write_mastery_observations``，strength 强化也在那里：
    good 观测 +1 封顶 15）。保留仅供历史参考/评测脚本，勿在新代码调用。

    与 append_mastery（LLM 开放抽取口径，历史兼容）的区别：id 由课程主题闭集给定，
    与门控读侧（proactive._read_mastery 的 topic_id 集）天然对齐，无断链。查不到 label
    的 hit（主题已删）跳过。affirm 信号 → strength += 1（MemoryBank spacing effect）。
    无 delta 的中性命中也记一次观测（observation_count++，喂后续 repeat 判定）。
    """
    from core.db.database import CourseTopic, KnowledgeMastery
    from sqlalchemy import select

    if not hits:
        return 0
    tids = [tid for tid, _sim in hits]
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

    risk_delta = _delta(signal.get("risk_delta"))
    mastery_delta = _delta(signal.get("mastery_delta"))
    affirm = "affirm" in (signal.get("signals") or [])
    now = time.time()
    n = 0
    for tid, _sim in hits:
        label = label_map.get(tid)
        if not label:
            continue
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
        if row is not None:
            row.mastery = _clamp(row.mastery + mastery_delta, 0.5)
            row.risk = _clamp(row.risk + risk_delta, 0.5)
            row.observation_count = (row.observation_count or 1) + 1
            row.last_observed_at = now
            if affirm:
                row.strength = (row.strength or 5.0) + 1.0
            ev = list(row.evidence_episode_ids or [])
            if episode_id and episode_id not in ev:
                ev.append(episode_id)
            row.evidence_episode_ids = ev[-20:]
        else:
            db.add(
                KnowledgeMastery(
                    user_id=user_id,
                    course_id=course_id,
                    kp_id=tid,
                    label=label,
                    mastery=_clamp(0.5 + mastery_delta, 0.5),
                    risk=_clamp(0.5 + risk_delta, 0.5),
                    observation_count=1,
                    first_observed_at=now,
                    last_observed_at=now,
                    evidence_episode_ids=[episode_id] if episode_id else [],
                    strength=6.0 if affirm else 5.0,
                )
            )
        n += 1
    await db.commit()
    return n


__all__ = [
    "append_mastery",
    "append_mastery_from_topics",
    "topic_repeat_hint",
    "get_weak_topics",
    "get_mastery_context",
    "forget_prob",
]


async def refit_class_difficulty(since_days: float = 3.0) -> int:
    """全班难度重折：近 since_days 天有新观测的 (course, topic)，取该主题全班全量
    观测跑 fold_class_difficulty，写回 course_topic.difficulty_b / n_item。返回重折主题数。

    自 worker.py 的 cron_refit_difficulty 内联 SQL 下沉（worker 只留观测+转调；
    fold 纯函数本就在 mastery_state.py）。口径不变：b 不在写路径更新（写侧判分观测
    只有错题本一处，对话为主的流量下在线重折会让 b 恒 0）；每个学生各自 theta 对上
    共享 b 成对更新（Pelánek/Klinkenberg 口径），按来源权重吃全部观测（判分 1.0 /
    对话 0.6 / 词表 0.3）。全量重折天然幂等；不回写学生 theta（活跃学生下次巩固
    全量重折自动切到新 b）。零 LLM。
    """
    import time as _time

    from sqlalchemy import select, update

    from core.db.database import AsyncSessionLocal, CourseTopic, MasteryObservation
    from core.memory.mastery_state import fold_class_difficulty

    since = _time.time() - since_days * 86400
    async with AsyncSessionLocal() as db:
        pairs = (
            await db.execute(
                select(
                    MasteryObservation.course_id, MasteryObservation.topic_id
                ).where(
                    MasteryObservation.created_at >= since,
                    MasteryObservation.course_id != "",
                ).distinct()
            )
        ).all()
        for cid, tid in pairs:
            obs = (
                (
                    await db.execute(
                        select(
                            MasteryObservation.user_id,
                            MasteryObservation.outcome,
                            MasteryObservation.weight,
                            MasteryObservation.created_at,
                        ).where(
                            MasteryObservation.course_id == cid,
                            MasteryObservation.topic_id == tid,
                        )
                    )
                )
                .mappings()
                .all()
            )
            b, n_item = fold_class_difficulty(list(obs))
            await db.execute(
                update(CourseTopic)
                .where(
                    CourseTopic.course_id == cid,
                    CourseTopic.topic_id == tid,
                )
                .values(difficulty_b=round(b, 6), n_item=n_item)
            )
        await db.commit()
    return len(pairs)
