"""学情拼接门控（主创新）：跨会话学情从每轮无条件注入，改为代价敏感过闸才塞。

- ``decide_stitch``：纯函数决策核心（可单测、可向教师解释"凭啥拼"），由
  ``scripts.eval_memory.stitch_cases`` 做 TDD（9 个正负例 should 决策全过）。
- ``stitch_for_turn``：运行时包装——算 S_t（问句 embedding 最近邻）+ 读 mastery/error
  + 调 decide_stitch，返回 ``StitchBrief``（含拼入 prompt 的简报文本或空）。

接入 ``pipeline_common.build_common_context_layers`` 替换无条件 ``get_mastery_context``：
门控说不拼则 ``mastery_context`` 为空（沉默 = 这轮不带跨会话预警，不是系统没回）。

数据层约束（id-align）：门控**只读** ``knowledge_mastery``（经 ``course_topic_store``
对齐 topic_id；risk/logit 由 mastery_state 折叠状态派生）与 ``mastery_observation``
（C2 复发 = 同主题 poor 计数 ≥ 2，30 天窗）。**禁止**读 ``users.knowledge_graph`` /
``users.error_graph``——那是仪表盘展示层，非门控数据源（error_graph 的自由标签哈希
节点数不准「同一个错又犯」，见 plan §3f）。

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
# 读侧 unknown 判定阈值（余弦）：独立常量，**不得**与写侧 match_topics 的
# TOPIC_MATCH_THRESHOLD 合并单源——写侧判定召回的阈值放宽（plan §4a-2，只作用写侧）
# 时，读侧若跟着降阈会把无关问句错认成主题，直接放大门控误报。两常量必须分开。
UNKNOWN_THRESHOLD = 0.55

# C1 入闸判据（plan §3e）：logit 空间 theta - b_topic <= WEAK_MARGIN。
# 语义锚点 =「同一主题至少 2 次判 poor」（Elo 折叠下 1 次 poor ≈ -0.30、2 次 ≈ -0.54，
# 本值取 logit(0.4) ≈ -0.4055，恰等价 risk >= 0.6，便于换算成人能读的概率）。
# 纪律：不写死就算完——test_mastery_state 由 fold_observations 实算夹出此值须落在
# 1 次与 2 次 poor 之间；W_JUDGE 变了测试会红，强制重定常量。
WEAK_MARGIN = -0.4055

# C2 复发：观测日志中同主题 poor 计数（30 天窗，与 p_forget S0=5 量级相称，进 eval 扫）
RECURRENCE_WINDOW_DAYS = 30

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
    severity_map: dict[str, float] | None = None,
    now: float | None = None,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    weak_margin: float = WEAK_MARGIN,
) -> StitchBrief:
    """纯函数门控决策（VoI 三因子版，零 LLM）。所有输入已对齐 topic_id 口径。

    Args:
        matched_topic: 问句命中的主题（S_t），None = unknown。
        prereq_closure: matched 的全部前置主题（course_topic_store.get_prereq_predecessors 算）。
        mastery_map: {topic_id: {risk, risk_used, p_forget, logit, label, observation_count}}，
            门控唯一学情数据源。logit = theta - b_topic（折叠状态，读侧不再收缩）。
        error_repeated: 反复出错的 topic_id 集（mastery_observation 同主题 poor 计数 ≥2 派生）。
        confidence: 问句→主题匹配的置信度 c ∈ [0,1]（sim 经 (sim-θ)/(1-θ) 归一；Ambiguity 因子）。
        severity_map: {topic_id: safety_weight}，Task Risk 因子按**证据主题**（真正预警的
            前置/复发主题）取——不是问句命中的主题（学生问基尔霍夫、薄弱的是通电操作
            前置时，该加权的是后者）。缺省 1.0 = 不加权一致。
        now: cooldown 判定的当前时间戳（调用方传入，保持纯函数可单测；None=用 time.time）。
        冷却时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带，见 _read_mastery）。
        weak_margin: C1 入闸的 logit 边际（plan §3e，见模块头 WEAK_MARGIN 注释）。

    候选（只在两类上算 p_need，全库薄弱点禁止入闸）：
        C1 未问也带 = (prereq_closure - {matched}) ∩ {logit <= weak_margin}
        C2 问到了但提旧错 = matched ∈ error_repeated

    决策（VoI 三因子 + 三档）：证据按 score = p_need·severity 选取（value 去掉 c 的
    口径，危险前置可越过 p_need 更高的平淡候选），档位按 value = p_need · severity · c
    取 NetVoI 最大的动作：
        full   = 1.0·value - 0.30
        light  = 0.6·value - 0.12
        silent = 0
    等价边界（教师可理解口径）：value<0.20 沉默 / 0.20≤value<0.45 轻提醒 / ≥0.45 完整。
    """
    # 1. unknown 直接不拼（问句没命中任何课程主题）
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    # 2. C1 前置缺口：前置闭包（不含 matched）里薄弱的（logit 空间判据，plan §3e：
    #    比 risk 不比被 sigmoid 压扁的概率；「至少两次判 poor」的语义锚点），
    #    按薄弱程度（logit 升序 = risk 降序）排
    prereq_gaps = sorted(
        [
            (tid, mastery_map[tid])
            for tid in (prereq_closure - {matched_topic})
            if tid in mastery_map
            and mastery_map[tid].get("logit", 0.0) <= weak_margin
        ],
        key=lambda x: x[1].get("logit", 0.0),
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
    # 证据选取按 score = p_need·sev（与第 6 步 value 去掉 c 的口径一致），不能只比
    # p_need——否则 sev 只能影响已选中证据的档位，p_need 略低但安全权重高的危险
    # 前置永远抢不过平淡的复发候选
    def _sev(t: str) -> float:
        return max(0.0, (severity_map or {}).get(t, 1.0))

    score = 0.0
    if prereq_gaps:
        top_tid, top = max(prereq_gaps, key=lambda x: _need_gap(x[1]) * _sev(x[0]))
        p_need, evidence, reason, sev = _need_gap(top), top_tid, "prereq_gap", _sev(top_tid)
        score = p_need * sev
    if is_recurrence:
        rec = max(
            RECURRENCE_PNEED_FLOOR,
            mastery_map.get(matched_topic, {}).get("risk_used", 0),
        )
        rec_score = rec * _sev(matched_topic)
        if rec_score > score:
            p_need, evidence, reason, sev = rec, matched_topic, "recurrence", _sev(matched_topic)

    # 5. cooldown：证据主题 24h 内实际拼过 → 直接沉默（防过载；AIED N=81 依据）。
    #    时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带）。
    if evidence is not None:
        ls = mastery_map.get(evidence, {}).get("last_stitched_at")
        if ls is not None:
            now_t = now if now is not None else time.time()
            if (now_t - (ls or 0)) < COOLDOWN_HOURS * 3600:
                return StitchBrief(False, None, "cooldown", p_need, 0.0)

    # 6. VoI 三档：value = p_need·severity·c（Task Risk × Ambiguity），NetVoI 最大者胜。
    #    severity 沿用第 4 步胜出证据的取值（选取与打分同一口径，单一真相源），
    #    按**证据主题**取（预警指向的薄弱前置/复发主题），非问句命中主题。
    c = min(1.0, max(0.0, confidence))
    value = p_need * sev * c
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
    weak_margin: float = WEAK_MARGIN,
) -> StitchBrief:
    """PRISM 两参数单阈值门控（对照臂，供 eval 消融）。语义与 decide_stitch 同候选口径。

    τ = C_FA / (C_FA + p_need·C_FN)，p_need >= τ 才拼（单边界 full-or-silent）。
    p_need 计算与 VoI 版共享同一套候选（logit 判据）与遗忘口径（复制主体，非互相
    调用——两臂在 eval 中必须独立可审计）。
    """
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    prereq_gaps = sorted(
        [
            (tid, mastery_map[tid])
            for tid in (prereq_closure - {matched_topic})
            if tid in mastery_map and mastery_map[tid].get("logit", 0.0) <= weak_margin
        ],
        key=lambda x: x[1].get("logit", 0.0),
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

    # 5. 决策（VoI：confidence 来自最近邻 sim，severity 按证据主题取——闭包+matched
    #    的 safety_weight 一次查齐传入，decide_stitch 内按 evidence 选取。evidence
    #    只可能是闭包主题或 matched 本身，error_repeated 里其余主题查了也用不上）
    severity_map = await _severity_map({matched_topic} | closure, course_id, db)
    brief = decide_stitch(
        matched_topic, closure, mastery_map, error_repeated,
        confidence=confidence, severity_map=severity_map,
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


async def _severity_map(topic_ids: set[str], course_id: str, db) -> dict[str, float]:
    """候选主题的 course_topic.safety_weight 批量读（Task Risk 因子，按证据主题取）。

    列原名 severity，与 error_patterns.severity（错误严重度）撞义已改名 safety_weight。
    一次查齐整组候选，免逐主题 round-trip；缺行按默认 1.0（=不加权一致）。
    """
    from core.db.database import CourseTopic
    from sqlalchemy import select

    tids = [t for t in topic_ids if t]
    if not tids:
        return {}
    rows = (
        await db.execute(
            select(CourseTopic.topic_id, CourseTopic.safety_weight).where(
                CourseTopic.course_id == course_id,
                CourseTopic.topic_id.in_(tids),
            )
        )
    ).all()
    # w=0 是合法教师标注（「永不因它预警」），不能被 `w or 1.0` 洗成默认 1.0
    return {t: float(w) if w is not None else 1.0 for t, w in rows}


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
    """读 knowledge_mastery → {topic_id: {risk, risk_used, p_forget, logit, label, ...}}。

    id-align：行 kp_id 非 topic_id 时，用 label 经 topic_map（label_norm→topic_id）本地兜底，
    避免 N+1（topic_map 由 stitch_for_turn 一次取全传入）。

    掌握度估计器重构（阶段D）：
    - risk 由折叠状态算出（mastery = sigmoid(theta - b_topic)，risk = 1 - mastery），
      b_topic 读 course_topic.difficulty_b（全班共享，题目难度即班级先验）；
    - k=3 班级收缩已删（与 U(n) 和 b_topic 先验功能重复且更粗），risk_used = risk；
    - logit = theta - b_topic 一并放进 map——C1 判据在 logit 空间（WEAK_MARGIN），
      没被 sigmoid 压扁；
    - 遗忘口径（MemoryBank）不变：p_forget = 1 - exp(-t/S)。
    """
    from core.db.database import CourseTopic, KnowledgeMastery
    from core.memory.course_topic_store import _norm_label
    from core.memory.mastery import forget_prob
    from core.memory.mastery_state import sigmoid
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
    # 题目难度（全班共享）一次取齐，行内不存（单一真相源在 course_topic）
    b_map: dict[str, float] = {}
    tids_hint = {r.kp_id for r in rows if r.kp_id}
    if tids_hint:
        b_rows = (
            await db.execute(
                select(CourseTopic.topic_id, CourseTopic.difficulty_b).where(
                    CourseTopic.course_id == course_id,
                    CourseTopic.topic_id.in_(tids_hint),
                )
            )
        ).all()
        b_map = {t: float(b or 0.0) for t, b in b_rows}

    now = time.time()
    out: dict[str, dict] = {}
    # id-align：只信任落在 course_topic.topic_id 集内的 kp_id，其余走 label 归一化对齐。
    topic_ids = set(topic_map.values())
    for r in rows:
        tid = r.kp_id if r.kp_id in topic_ids else topic_map.get(_norm_label(r.label))
        if not tid:
            continue
        b = b_map.get(tid, 0.0)
        # risk 按折叠状态重算（不读旧列——旧行可能是 delta 累加口径的历史值）；
        # theta 为 0 且无观测的行（历史遗留）等于无信息先验 risk=0.5
        logit = (r.theta or 0.0) - b
        risk = 1.0 - sigmoid(logit)
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        out[tid] = {
            "risk": round(risk, 4),
            "risk_used": round(risk, 4),  # 收缩已删（b_topic 即班级先验），保留键供门控口径不变
            "logit": round(logit, 4),
            "p_forget": forget_prob(age_days, r.strength or 5.0),
            "strength": r.strength or 5.0,
            "label": r.label,
            "observation_count": r.observation_count or 1,
            "last_stitched_at": r.last_stitched_at,
        }
    return out


async def _read_error_repeated(
    user_id: str, course_id: str, db, topic_map: dict[str, str]
) -> set[str]:
    """C2 复发：观测日志中同主题 poor 计数 ≥ 2（30 天窗）派生（plan §3f）。

    不再读 users.error_graph——节点 id 是对 LLM 自由标签哈希的，同一错误换个说法
    就是新节点，error_count 永远 1；且门控读仪表盘数据违反数据层约束。观测日志的
    topic_id 是闭集固定 id，数得准；source 不过滤（聊天判 poor + 答错都算复发，
    跨证据类型统一）。error_graph 退回纯仪表盘用途。
    """
    from core.db.database import MasteryObservation
    from core.memory.mastery_state import POOR_OUTCOME_MAX
    from sqlalchemy import func, select

    cutoff = time.time() - RECURRENCE_WINDOW_DAYS * 86400
    rows = (
        await db.execute(
            select(MasteryObservation.topic_id, func.count())
            .where(
                MasteryObservation.user_id == user_id,
                MasteryObservation.course_id == course_id,
                MasteryObservation.outcome <= POOR_OUTCOME_MAX,
                MasteryObservation.created_at >= cutoff,
            )
            .group_by(MasteryObservation.topic_id)
            .having(func.count() >= 2)
        )
    ).all()
    valid = set(topic_map.values())
    return {tid for tid, _cnt in rows if tid in valid}


__all__ = ["StitchBrief", "decide_stitch", "decide_stitch_prism", "stitch_for_turn"]
