"""学情拼接门控（主创新）：跨会话学情从每轮无条件注入，改为代价敏感过闸才塞。

- ``decide_stitch``：纯函数决策核心（可单测、可向教师解释"凭啥拼"），由
  ``scripts.eval_memory.stitch_cases`` 做 TDD（9 个正负例 should 决策全过）。
- ``stitch_for_turn``：运行时包装——算 S_t（问句 embedding 最近邻）+ 读 mastery/error
  + 调 decide_stitch，返回 ``StitchBrief``（含拼入 prompt 的简报文本或空）。

接入 ``pipeline_common.build_common_context_layers`` 替换无条件 ``get_mastery_context``：
门控说不拼则 ``mastery_context`` 为空（沉默 = 这轮不带跨会话预警，不是系统没回）。

数据层约束（id-align）：门控**只读** ``knowledge_mastery``（经 ``course_topic_store``
对齐 topic_id）；error ``repeated`` 从 ``users.error_graph`` 读。**禁止**读
``users.knowledge_graph`` 节点上的 risk/mastery（那是仪表盘展示层，非门控数据源）。

代价公式借 PRISM（ICLR 2026）：``p_need >= τ``，``τ = C_FA / (C_FA + p_need·C_FN)``，
实验课默认 ``C_FN > C_FA``（漏报 = 下次还错可能损坏仪器，重于误报 = 答非所问）。
首版不估 ``p_accept``（无弹窗式干预，无"用户接不接受"）。输出 ``should_stitch=False``
即显式不拼（PRISM 的 null-intervention，但是拼不拼学情，不是发不发消息）。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- 门控参数（可调，代价扫描在 eval-c 跑）----
from core.memory.course_topic_store import TOPIC_MATCH_THRESHOLD as UNKNOWN_THRESHOLD  # noqa: F401
# ↑ 阈值单源：读侧 unknown 判定与写侧 match_topics 命中过滤共用 course_topic_store
#   的 TOPIC_MATCH_THRESHOLD，避免两处常量漂移出不同口径。

RISK_THRESHOLD = 0.5          # mastery risk >= 此值视为薄弱（与 get_mastery_context 同口径）
DEFAULT_C_FA = 1.0
DEFAULT_C_FN = 3.0            # 漏报 3 倍代价（实验课损坏仪器风险 > 答非所问）
RECURRENCE_PNEED_FLOOR = 0.7  # 复发（C2）固定需求下限

# ---- VoI 三因子（阶段6）：确定性效用表，非 parameter-free（论文口径须如实声明）----
L_FULL = 0.30                 # 完整简报的认知负荷代价
L_LIGHT = 0.12                # 一句轻提醒的代价
L_SILENT = 0.0
V_LIGHT_COEF = 0.6            # 轻提醒的实现价值系数（完整简报=1.0）
COOLDOWN_HOURS = 24           # 同一证据主题 24h 内不重复拼（防过载，AIED N=81 依据）


@dataclass(frozen=True)
class StitchBrief:
    """门控决策结果。should_stitch=False 即显式不拼。

    level（VoI 三档）：silent=不拼 / light=一句轻提醒 / full=完整简报。
    should_stitch ≡ level != "silent"，pipeline_common 等调用方零改动。
    """

    should_stitch: bool
    evidence_topic: str | None          # 该拼时，简报指向的证据主题（topic_id）
    reason: str                         # prereq_gap / recurrence / unknown / no_candidate / below_value / cooldown
    p_need: float
    tau: float
    text: str = ""                      # 应拼时填：拼入 prompt 的简报文本
    level: str = "silent"               # silent / light / full（PRISM 对照臂恒 full 或 silent）


def decide_stitch(
    matched_topic: str | None,
    prereq_closure: set[str],
    mastery_map: dict[str, dict],
    error_repeated: set[str],
    *,
    confidence: float = 1.0,
    severity: float = 1.0,
    now: float | None = None,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    risk_threshold: float = RISK_THRESHOLD,
) -> StitchBrief:
    """纯函数门控决策（VoI 三因子版，零 LLM）。所有输入已对齐 topic_id 口径。

    Args:
        matched_topic: 问句命中的主题（S_t），None = unknown。
        prereq_closure: matched 的全部前置主题（course_topic_store.get_prereq_predecessors 算）。
        mastery_map: {topic_id: {risk, risk_used, p_forget, label, observation_count}}，门控唯一学情数据源。
        error_repeated: 反复出错的 topic_id 集（来自 error_graph repeated，label→topic_id 对齐）。
        confidence: 问句→主题匹配的置信度 c ∈ [0,1]（sim 经 (sim-θ)/(1-θ) 归一；Ambiguity 因子）。
        severity: 证据主题的 course_topic.severity（Task Risk 因子；默认 1.0 行为与不加一致）。
        now: cooldown 判定的当前时间戳（调用方传入，保持纯函数可单测；None=用 time.time）。
        冷却时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带，见 _read_mastery）。

    候选（只在两类上算 p_need，全库薄弱点禁止入闸）：
        C1 未问也带 = (prereq_closure - {matched}) ∩ {risk >= risk_threshold}
        C2 问到了但提旧错 = matched ∈ error_repeated

    决策（VoI 三因子 + 三档）：value = p_need · severity · c，取 NetVoI 最大的动作：
        full   = 1.0·value - 0.30
        light  = 0.6·value - 0.12
        silent = 0
    等价边界（教师可理解口径）：value<0.20 沉默 / 0.20≤value<0.45 轻提醒 / ≥0.45 完整。
    """
    # 1. unknown 直接不拼（问句没命中任何课程主题）
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    # 2. C1 前置缺口：前置闭包（不含 matched）里高 risk 的，按 risk 降序
    prereq_gaps = sorted(
        [
            (tid, mastery_map[tid])
            for tid in (prereq_closure - {matched_topic})
            if tid in mastery_map and mastery_map[tid].get("risk", 0) >= risk_threshold
        ],
        key=lambda x: -x[1].get("risk", 0),
    )
    # 3. C2 复发：matched 本身反复出错
    is_recurrence = matched_topic in error_repeated

    if not prereq_gaps and not is_recurrence:
        # 无候选：前置不缺口、当前问的也没复发 → 沉默（不把无关薄弱点塞进来）
        return StitchBrief(False, None, "no_candidate", 0.0, 1.0)

    # 4. p_need（阶段3 遗忘耦合，MemoryBank 口径）：
    #    C1 前置缺口 = risk_used · p_forget——薄弱且久未触碰才值得拼；刚观测过的
    #    薄弱点（p_forget≈0）不用预警（现在正学着呢）。C2 复发不乘遗忘（复发是错题
    #    信号不是时间信号），保持 max(0.7, risk_used)。
    def _need_gap(m: dict) -> float:
        return m.get("risk_used", m.get("risk", 0)) * m.get("p_forget", 1.0)

    p_need = 0.0
    evidence = None
    reason = ""
    if prereq_gaps:
        top_tid, top = max(prereq_gaps, key=lambda x: _need_gap(x[1]))
        p_need, evidence, reason = _need_gap(top), top_tid, "prereq_gap"
    if is_recurrence:
        rec = max(
            RECURRENCE_PNEED_FLOOR,
            mastery_map.get(matched_topic, {}).get("risk_used", 0),
        )
        if rec > p_need:
            p_need, evidence, reason = rec, matched_topic, "recurrence"

    # 5. cooldown：证据主题 24h 内实际拼过 → 直接沉默（防过载；AIED N=81 依据）。
    #    时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带）。
    if evidence is not None:
        ls = mastery_map.get(evidence, {}).get("last_stitched_at")
        if ls is not None:
            now_t = now if now is not None else time.time()
            if (now_t - (ls or 0)) < COOLDOWN_HOURS * 3600:
                return StitchBrief(False, None, "cooldown", p_need, 0.0)

    # 6. VoI 三档：value = p_need·severity·c（Task Risk × Ambiguity），NetVoI 最大者胜
    c = min(1.0, max(0.0, confidence))
    value = p_need * max(0.0, severity) * c
    net_full = 1.0 * value - L_FULL
    net_light = V_LIGHT_COEF * value - L_LIGHT
    if net_full <= 0 and net_light <= 0:
        return StitchBrief(False, None, "below_value", p_need, 0.0)
    if net_full >= net_light:
        return StitchBrief(True, evidence, reason, p_need, 0.0, level="full")
    return StitchBrief(True, evidence, reason, p_need, 0.0, level="light")


def decide_stitch_prism(
    matched_topic: str | None,
    prereq_closure: set[str],
    mastery_map: dict[str, dict],
    error_repeated: set[str],
    *,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    risk_threshold: float = RISK_THRESHOLD,
) -> StitchBrief:
    """PRISM 两参数单阈值门控（对照臂，供 eval 消融）。语义与旧 decide_stitch 完全一致。

    τ = C_FA / (C_FA + p_need·C_FN)，p_need >= τ 才拼（单边界 full-or-silent）。
    p_need 计算与 VoI 版共享同一套候选与遗忘口径（复制主体，非互相调用——两臂在
    eval 中必须独立可审计）。
    """
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    prereq_gaps = sorted(
        [
            (tid, mastery_map[tid])
            for tid in (prereq_closure - {matched_topic})
            if tid in mastery_map and mastery_map[tid].get("risk", 0) >= risk_threshold
        ],
        key=lambda x: -x[1].get("risk", 0),
    )
    is_recurrence = matched_topic in error_repeated

    if not prereq_gaps and not is_recurrence:
        return StitchBrief(False, None, "no_candidate", 0.0, 1.0)

    def _need_gap(m: dict) -> float:
        return m.get("risk_used", m.get("risk", 0)) * m.get("p_forget", 1.0)

    p_need = 0.0
    evidence = None
    reason = ""
    if prereq_gaps:
        top_tid, top = max(prereq_gaps, key=lambda x: _need_gap(x[1]))
        p_need, evidence, reason = _need_gap(top), top_tid, "prereq_gap"
    if is_recurrence:
        rec = max(
            RECURRENCE_PNEED_FLOOR,
            mastery_map.get(matched_topic, {}).get("risk_used", 0),
        )
        if rec > p_need:
            p_need, evidence, reason = rec, matched_topic, "recurrence"

    tau = c_fa / (c_fa + p_need * c_fn)
    should = p_need >= tau
    return StitchBrief(
        should, evidence if should else None, reason if should else "below_tau", p_need, tau,
        level="full" if should else "silent",
    )


async def stitch_for_turn(
    query: str, user_id: str, course_id: str, db
) -> StitchBrief:
    """运行时包装：算 S_t + 读 mastery/error + 调 decide_stitch。

    embed_model 在内部解析（get_embed_model 单例，失败 fail-safe 不拼），调用方无需关心。
    matched_topic 经 embedding 最近邻；无 embedding（course_topic 未灌 embedding 或不可用）
    → fail-safe 返回不拼，不阻塞回答。
    """
    from dataclasses import replace

    from core.memory.course_topic_store import get_course_topic_map, get_prereq_predecessors

    # 0. embed_model（单例；不可用 → 门控 fail-safe 不拼）
    embed_model = None
    try:
        from core.rag.llamaindex.pg_store import get_embed_model
        embed_model = get_embed_model()
    except Exception:
        pass

    # 1. S_t 问句最近邻（含 sim → VoI 的 confidence 因子）
    hit = await _match_topic(query, course_id, db, embed_model)
    if hit is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)
    matched_topic, matched_sim = hit
    # confidence 归一：sim 刚过阈值 → 0（含糊，预警可能指错前置），sim→1 → 1（笃定）
    theta = UNKNOWN_THRESHOLD
    confidence = min(1.0, max(0.0, (matched_sim - theta) / (1.0 - theta))) if theta < 1 else 1.0

    # 2. 前置闭包 + label→topic_id 映射（各一次查询，下游复用免 N+1）
    closure = await get_prereq_predecessors(matched_topic, course_id, db)
    topic_map = await get_course_topic_map(course_id, db)

    # 3/4. mastery + error repeated（topic_map 本地对齐，无逐行 DB 查询）
    mastery_map = await _read_mastery(user_id, course_id, db, topic_map)
    error_repeated = await _read_error_repeated(user_id, course_id, db, topic_map)

    # 5. 决策（VoI：confidence 来自最近邻 sim，severity 来自证据主题行）
    severity = await _topic_severity(matched_topic, course_id, db)
    brief = decide_stitch(
        matched_topic, closure, mastery_map, error_repeated,
        confidence=confidence, severity=severity,
    )

    # 6. 应拼则填简报文本（frozen dataclass 用 replace，免手工抄字段）+ 回写 cooldown 戳
    if brief.should_stitch and brief.evidence_topic:
        ev = mastery_map.get(brief.evidence_topic, {})
        label = ev.get("label") or brief.evidence_topic
        await _mark_stitched(db, user_id, course_id, brief.evidence_topic)
        if brief.level == "light":
            text = f"（顺带一提：该生「{label}」此前偏薄弱，可留意）"
        else:
            text = (
                f"## 学情提示（跨会话，未问亦带）\n该生在「{label}」偏薄弱"
                f"（{ev.get('observation_count', '?')}次观测），本轮涉及其前置/相关内容，请留意诊断。"
            )
        return replace(brief, text=text)
    return brief


async def _topic_severity(topic_id: str, course_id: str, db) -> float:
    """证据主题的 course_topic.severity（Task Risk 因子）。默认 1.0（=行为与不加列一致）。"""
    from core.db.database import CourseTopic
    from sqlalchemy import select

    val = (
        await db.execute(
            select(CourseTopic.severity).where(
                CourseTopic.course_id == course_id,
                CourseTopic.topic_id == topic_id,
            )
        )
    ).scalar()
    return float(val) if val else 1.0


async def _mark_stitched(db, user_id: str, course_id: str, topic_id: str) -> None:
    """实际拼出简报后回写 last_stitched_at（cooldown 的写入端）。失败非致命。"""
    from core.db.database import KnowledgeMastery
    from sqlalchemy import update

    try:
        await db.execute(
            update(KnowledgeMastery)
            .where(
                KnowledgeMastery.user_id == user_id,
                KnowledgeMastery.course_id == course_id,
                KnowledgeMastery.kp_id == topic_id,
            )
            .values(last_stitched_at=time.time())
        )
        await db.commit()
    except Exception as exc:
        logger.warning("[stitch] 回写 last_stitched_at 失败（非致命）：%s", exc)
        await db.rollback()


async def _match_topic(query: str, course_id: str, db, embed_model) -> tuple[str, float] | None:
    """S_t：query embedding vs course_topic.embedding 余弦最近邻。

    返回 (topic_id, sim)——sim 供 VoI 的 confidence 因子（(sim-θ)/(1-θ) 归一在调用方）；
    低于阈值 = unknown = None。
    """
    from core.db.database import CourseTopic
    from core.memory.course_topic_store import cosine
    from sqlalchemy import select

    if embed_model is None:
        return None
    rows = (
        (await db.execute(select(CourseTopic).where(CourseTopic.course_id == course_id)))
        .scalars()
        .all()
    )
    candidates = [(r.topic_id, r.embedding) for r in rows if r.embedding]
    if not candidates:
        return None
    try:
        q_emb = await embed_model._aget_query_embedding(query)
    except Exception as exc:
        logger.warning("[stitch] query embedding 失败（fail-safe 不拼）：%s", exc)
        return None

    best_tid, best_sim = max(
        ((tid, cosine(q_emb, emb)) for tid, emb in candidates), key=lambda x: x[1]
    )
    return (best_tid, best_sim) if best_sim >= UNKNOWN_THRESHOLD else None


async def _read_mastery(
    user_id: str, course_id: str, db, topic_map: dict[str, str]
) -> dict[str, dict]:
    """读 knowledge_mastery → {topic_id: {risk, risk_used, p_forget, label, ...}}。

    id-align：行 kp_id 非 topic_id 时，用 label 经 topic_map（label_norm→topic_id）本地兜底，
    避免 N+1（topic_map 由 stitch_for_turn 一次取全传入）。

    遗忘口径（阶段3，MemoryBank）：p_forget = 1 - exp(-t/S)，S 取行内 strength。
    旧 eff_risk = risk·exp(-λt) 已废——方向反了：时间越久越该拼前置提醒，不是越不拼。
    λ=0.01 仅保留在 get_mastery_context 的排序里，门控不再用。
    risk_used 为阶段4 班级收缩的落点，当前（收缩未启用时）= 个人 risk。
    """
    from core.db.database import KnowledgeMastery
    from core.memory.course_topic_store import _norm_label
    from core.memory.mastery import forget_prob
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
    now = time.time()
    out: dict[str, dict] = {}
    # id-align：kp_id 恒非空（写侧 entity_id 或 label 哈希兜底），`or` 短路令 label 兜底
    # 永不生效。只信任落在 course_topic.topic_id 集内的 kp_id，其余走 label 归一化对齐。
    topic_ids = set(topic_map.values())
    for r in rows:
        tid = r.kp_id if r.kp_id in topic_ids else topic_map.get(_norm_label(r.label))
        if not tid:
            continue
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        out[tid] = {
            "risk": r.risk or 0,
            "risk_used": r.risk or 0,  # 下方班级收缩覆写
            "p_forget": forget_prob(age_days, r.strength or 5.0),
            "strength": r.strength or 5.0,
            "label": r.label,
            "observation_count": r.observation_count or 1,
            "last_stitched_at": r.last_stitched_at,
        }

    # 班级收缩（阶段4，经验贝叶斯）：观测少的主题向全班均值靠——新生 n=1 的偶发高
    # risk 不至于直接触发预警；n 足够大时几乎全信个人。k=3 平滑常数。
    # 该 topic 全班只有自己观测过时 cohort_mean=个人值，公式退化成个人 risk。
    if out:
        from sqlalchemy import func

        cohort_rows = (
            await db.execute(
                select(
                    KnowledgeMastery.kp_id,
                    func.avg(KnowledgeMastery.risk),
                ).where(
                    KnowledgeMastery.course_id == course_id,
                    KnowledgeMastery.kp_id.in_(list(out)),
                ).group_by(KnowledgeMastery.kp_id)
            )
        ).all()
        k = 3
        for tid, cohort_mean in cohort_rows:
            m = out[tid]
            n = m["observation_count"]
            mean = float(cohort_mean) if cohort_mean is not None else m["risk"]
            m["risk_used"] = round(
                (n / (n + k)) * m["risk"] + (k / (n + k)) * mean, 4
            )
    return out


async def _read_error_repeated(
    user_id: str, course_id: str, db, topic_map: dict[str, str]
) -> set[str]:
    """从 users.error_graph 读 repeated/error_count>1 的 topic，label→topic_id 本地对齐。"""
    from core.db.database import User
    from core.memory.course_topic_store import _norm_label
    from sqlalchemy import select

    row = (await db.execute(select(User.error_graph).where(User.id == user_id))).first()
    eg = row.error_graph if row else None
    if not isinstance(eg, dict):
        return set()
    out: set[str] = set()
    for n in eg.get("nodes") or []:
        if n.get("repeated") or (n.get("error_count") or 0) > 1:
            tid = topic_map.get(_norm_label(n.get("label") or ""))
            if tid:
                out.add(tid)
    return out


__all__ = ["StitchBrief", "decide_stitch", "decide_stitch_prism", "stitch_for_turn"]
