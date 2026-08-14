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
import math
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- 门控参数（可调，代价扫描在 eval-c 跑）----
RISK_THRESHOLD = 0.5          # mastery risk >= 此值视为薄弱（与 get_mastery_context 同口径）
DEFAULT_C_FA = 1.0
DEFAULT_C_FN = 3.0            # 漏报 3 倍代价（实验课损坏仪器风险 > 答非所问）
RECURRENCE_PNEED_FLOOR = 0.7  # 复发（C2）固定需求下限
UNKNOWN_THRESHOLD = 0.55      # S_t 最近邻余弦低于此 = unknown = 不拼


@dataclass(frozen=True)
class StitchBrief:
    """门控决策结果。should_stitch=False 即显式不拼。"""

    should_stitch: bool
    evidence_topic: str | None          # 该拼时，简报指向的证据主题（topic_id）
    reason: str                         # prereq_gap / recurrence / unknown / no_candidate / below_tau
    p_need: float
    tau: float
    text: str = ""                      # 应拼时填：拼入 prompt 的简报文本


def decide_stitch(
    matched_topic: str | None,
    prereq_closure: set[str],
    mastery_map: dict[str, dict],
    error_repeated: set[str],
    *,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    risk_threshold: float = RISK_THRESHOLD,
) -> StitchBrief:
    """纯函数门控决策。所有输入已对齐 topic_id 口径。

    Args:
        matched_topic: 问句命中的主题（S_t），None = unknown。
        prereq_closure: matched 的全部前置主题（course_topic_store.get_prereq_predecessors 算）。
        mastery_map: {topic_id: {risk, eff_risk, label, observation_count}}，门控唯一学情数据源。
        error_repeated: 反复出错的 topic_id 集（来自 error_graph repeated，label→topic_id 对齐）。

    候选（只在两类上算 p_need，全库薄弱点禁止入闸）：
        C1 未问也带 = (prereq_closure - {matched}) ∩ {risk >= risk_threshold}
        C2 问到了但提旧错 = matched ∈ error_repeated
    """
    # 1. unknown 直接不拼（问句没命中任何课程主题）
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    # 2. C1 前置缺口：前置闭包（不含 matched）里高 risk 的，按风险降序
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

    # 4. p_need：取前置缺口最高 eff_risk 与复发加成的较大者
    def _eff(m: dict) -> float:
        return m.get("eff_risk", m.get("risk", 0))

    p_need = 0.0
    evidence = None
    reason = ""
    if prereq_gaps:
        top_tid, top = prereq_gaps[0]
        p_need, evidence, reason = _eff(top), top_tid, "prereq_gap"
    if is_recurrence:
        rec = max(RECURRENCE_PNEED_FLOOR, _eff(mastery_map.get(matched_topic, {})))
        if rec > p_need:
            p_need, evidence, reason = rec, matched_topic, "recurrence"

    # 5. τ 闸：p_need >= τ 才拼
    tau = c_fa / (c_fa + p_need * c_fn)
    should = p_need >= tau
    return StitchBrief(
        should, evidence if should else None, reason if should else "below_tau", p_need, tau
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

    # 1. S_t 问句最近邻
    matched_topic = await _match_topic(query, course_id, db, embed_model)
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    # 2. 前置闭包 + label→topic_id 映射（各一次查询，下游复用免 N+1）
    closure = await get_prereq_predecessors(matched_topic, course_id, db)
    topic_map = await get_course_topic_map(course_id, db)

    # 3/4. mastery + error repeated（topic_map 本地对齐，无逐行 DB 查询）
    mastery_map = await _read_mastery(user_id, course_id, db, topic_map)
    error_repeated = await _read_error_repeated(user_id, course_id, db, topic_map)

    # 5. 决策
    brief = decide_stitch(matched_topic, closure, mastery_map, error_repeated)

    # 6. 应拼则填简报文本（frozen dataclass 用 replace，免手工抄字段）
    if brief.should_stitch and brief.evidence_topic:
        ev = mastery_map.get(brief.evidence_topic, {})
        label = ev.get("label") or brief.evidence_topic
        return replace(
            brief,
            text=f"## 学情提示（跨会话，未问亦带）\n该生在「{label}」偏薄弱"
            f"（{ev.get('observation_count', '?')}次观测），本轮涉及其前置/相关内容，请留意诊断。",
        )
    return brief


async def _match_topic(query: str, course_id: str, db, embed_model) -> str | None:
    """S_t：query embedding vs course_topic.embedding 余弦最近邻；低于阈值 = unknown。"""
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
    return best_tid if best_sim >= UNKNOWN_THRESHOLD else None


async def _read_mastery(
    user_id: str, course_id: str, db, topic_map: dict[str, str]
) -> dict[str, dict]:
    """读 knowledge_mastery → {topic_id: {risk, eff_risk, label, observation_count}}。

    id-align：行 kp_id 非 topic_id 时，用 label 经 topic_map（label_norm→topic_id）本地兜底，
    避免 N+1（topic_map 由 stitch_for_turn 一次取全传入）。衰减用 mastery._DECAY_LAMBDA_PER_DAY
    单一常量，与 mastery.py 同源不漂移。
    """
    from core.db.database import KnowledgeMastery
    from core.memory.course_topic_store import _norm_label
    from core.memory.mastery import _DECAY_LAMBDA_PER_DAY
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
    for r in rows:
        tid = r.kp_id or topic_map.get(_norm_label(r.label))
        if not tid:
            continue
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        eff_risk = (r.risk or 0) * math.exp(-_DECAY_LAMBDA_PER_DAY * age_days)
        out[tid] = {
            "risk": r.risk or 0,
            "eff_risk": eff_risk,
            "label": r.label,
            "observation_count": r.observation_count or 1,
        }
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


__all__ = ["StitchBrief", "decide_stitch", "stitch_for_turn"]
