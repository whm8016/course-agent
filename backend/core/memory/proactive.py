"""学情拼接门控（主创新）：跨会话学情从每轮无条件注入，改为代价敏感过闸才塞。

- ``decide_stitch``：纯函数决策核心（可单测、可向教师解释"凭啥拼"），由
  ``scripts.eval_memory.stitch_cases`` 做 TDD（9 个正负例 should 决策全过）。
- ``stitch_for_turn``：运行时包装——算 S_t（问句 embedding 最近邻）+ 读 mastery/error
  + 调 decide_stitch，返回 ``StitchBrief``（含拼入 prompt 的简报文本或空）。

接入 ``pipeline_common.build_common_context_layers`` 替换无条件 ``get_mastery_context``：
门控说不拼则 ``mastery_context`` 为空（沉默 = 这轮不带跨会话预警，不是系统没回）。

数据层约束（id-align）：门控**只读** ``knowledge_mastery``（经 ``course_topic_store``
对齐 topic_id；risk/logit 由 mastery_state 折叠状态派生）与 ``mastery_observation``
（C2 复发 = 同主题 poor 计数 ≥ 1，30 天窗）。**禁止**读 ``users.knowledge_graph`` /
``users.error_graph``——那是仪表盘展示层，非门控数据源（error_graph 的自由标签哈希
节点数不准「同一个错又犯」，见 plan §3f）。

代价公式借 PRISM（ICLR 2026）：``p_need >= τ``，``τ = C_FA / (C_FA + p_need·C_FN)``，
实验课默认 ``C_FN > C_FA``（漏报 = 下次还错可能损坏仪器，重于误报 = 答非所问）。
首版不估 ``p_accept``（无弹窗式干预，无"用户接不接受"）。输出 ``should_stitch=False``
即显式不拼（PRISM 的 null-intervention，但是拼不拼学情，不是发不发消息）。

门控为**单阈值两档**（2026-08 由 VoI 三档简化定稿）：``p_need >= τ`` 过闸即一律
输出完整简报（不再分 light/full 档）。τ 由公式实算（单一真相源），默认代价
``C_FA=1 / C_FN=3`` 下代数正解恒为 ``p_need >= 0.434``——门槛不随轮变化；语义
锚点（「约六天未练」）由 test_stitch_gate 实算夹死。曾有的 value/severity/
confidence 乘子与 L_FULL/L_LIGHT 扣费表未经标定，已删（复发保底 0.7 > 0.434，
复发候选除 24h 冷却外必开口）。
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

# C1 入闸判据（plan §3e，口径改 theta）：theta <= WEAK_MARGIN。
# 语义锚点 =「同一主题至少 1 次判 poor」（2026-08-18 由 2 次下调：判定观测常摊到
# 兄弟主题上，凑两次等于把真缺口挡在门外；live 表E 12 个漏提里 ≈8 个死在这）。
# Elo 折叠（W_JUDGE=0.6）下 1 次 poor：b=0 → θ≈-0.30、b=0.5 → θ≈-0.2265，均 <= 本值；
# 本值取 logit(0.45) ≈ -0.2007，恰等价 risk >= 0.55，便于换算成人能读的概率。
# 不入闸的边界：fair-only（b∈[-0.5,0.5] 折完 θ >= -0.0735）、单次 W_RULE 观测
# （b=0 → -0.15）——低信源仍需两次；2 次 poor（≈-0.54）维持入闸不变。
# 不比 theta - b：b 是班级难度，一旦非零，一个 fair、theta≈0 的学生会被班级难度
# 直接拖过闸——把「该生薄弱」改写成「这题班里难」，与门控原意冲突。b 只进
# risk = 1 - sigmoid(theta - b)（影响已入闸候选的 p_need 排序），不决定谁能入闸。
# 纪律：不写死就算完——test_mastery_state 由 fold_observations 实算夹出此值须落在
# 1 次 poor 与 fair-only 之间；W_JUDGE 变了测试会红，强制重定常量。
WEAK_MARGIN = -0.2007

# C2 复发：观测日志中同主题 poor 计数（30 天窗，与 p_forget S0=5 量级相称，进 eval 扫）。
# 计数门槛 ≥1（与 WEAK_MARGIN 的「一次薄弱即管」同语义；2026-08-18 由 ≥2 下调）
RECURRENCE_WINDOW_DAYS = 30

DEFAULT_C_FA = 1.0
DEFAULT_C_FN = 3.0            # 漏报 3 倍代价（实验课损坏仪器风险 > 答非所问）
RECURRENCE_PNEED_FLOOR = 0.7  # 复发（C2）固定需求下限（必提，见模块头）
COOLDOWN_HOURS = 24           # 同一证据主题 24h 内不重复拼（防过载，AIED N=81 依据）


@dataclass(frozen=True)
class StitchBrief:
    """门控决策结果。should_stitch=False 即显式不拼。

    matched_topic：问句命中的主题（unknown 时 None）。pipeline_common 用它做第一路
    （memory_context）的主题过滤——两路分工：event 槽只注入本轮命中主题。
    """

    should_stitch: bool
    evidence_topic: str | None          # 该拼时，简报指向的证据主题（topic_id）
    reason: str                         # prereq_gap / recurrence / unknown / no_candidate / below_tau / cooldown
    p_need: float
    tau: float
    text: str = ""                      # 应拼时填：拼入 prompt 的简报文本
    matched_topic: str | None = None    # 问句命中的主题 id（第一路主题过滤用）


def decide_stitch(
    matched_topic: str | None,
    prereq_closure: set[str],
    mastery_map: dict[str, dict],
    error_repeated: set[str],
    *,
    now: float | None = None,
    c_fa: float = DEFAULT_C_FA,
    c_fn: float = DEFAULT_C_FN,
    weak_margin: float = WEAK_MARGIN,
) -> StitchBrief:
    """纯函数门控决策（单阈值两档，零 LLM）。所有输入已对齐 topic_id 口径。

    Args:
        matched_topic: 问句命中的主题（S_t），None = unknown。
        prereq_closure: matched 的全部前置主题（course_topic_store.get_prereq_predecessors 算）。
        mastery_map: {topic_id: {risk, p_forget, theta, logit, label, observation_count,
            safety_weight}}，门控唯一学情数据源。theta 是入闸判据（<= weak_margin）；
            logit = theta - b_topic 只供 risk/p_need 派生（班级难度不决定谁能入闸）。
            safety_weight 是教师标注的任务风险因子（默认 1.0），乘进 p_need——
            0=该主题永不因它预警、>1=更想预警（漏报代价更高，如通电操作类）。
        error_repeated: 反复出错的 topic_id 集（mastery_observation 同主题 poor 计数 ≥1 派生）。
        now: cooldown 判定的当前时间戳（调用方传入，保持纯函数可单测；None=用 time.time）。
            冷却时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带，见 _read_mastery）。
        c_fa / c_fn: 误报/漏报代价。τ 由公式实算（单一真相源），默认 1:3 下的
            门槛是恒定的 0.434（推导见模块头）。
        weak_margin: C1 入闸的 theta 边际（plan §3e，见模块头 WEAK_MARGIN 注释）。

    候选（只在两类上算 p_need，全库薄弱点禁止入闸）：
        C1 未问也带 = (prereq_closure - {matched}) ∩ {theta <= weak_margin}
        C2 问到了但提旧错 = matched ∈ error_repeated

    决策：证据按 max(p_need) 选取（复发臂与前置臂同口径竞争），p_need >= τ 过闸即
    输出完整简报（两档：过/不过，无 light 中间档）。p_need 钳 [0,1]（safety_weight
    乘进来可能超 1，τ 恒 <1，超 1 等价必过闸）。
    """
    # 1. unknown 直接不拼（问句没命中任何课程主题）
    if matched_topic is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)

    # 2. C1 前置缺口：前置闭包（不含 matched）里薄弱的（theta 判据，plan §3e：
    #    比 risk 不比被 sigmoid 压扁的概率；「至少一次判 poor」的语义锚点；不用
    #    theta - b——班级难度不决定谁能入闸，b 只影响入闸后的 risk/p_need）
    prereq_gaps = [
        (tid, mastery_map[tid])
        for tid in (prereq_closure - {matched_topic})
        if tid in mastery_map
        and mastery_map[tid].get("theta", 0.0) <= weak_margin
    ]
    # 3. C2 复发：matched 本身反复出错
    is_recurrence = matched_topic in error_repeated

    if not prereq_gaps and not is_recurrence:
        # 无候选：前置不缺口、当前问的也没复发 → 沉默（不把无关薄弱点塞进来）
        return StitchBrief(False, None, "no_candidate", 0.0, 1.0)

    # 4. p_need（阶段3 遗忘耦合，MemoryBank 口径）：
    #    C1 前置缺口 = risk · p_forget——薄弱且久未触碰才值得拼；刚观测过的
    #    薄弱点（p_forget≈0）不用预警（现在正学着呢）。C2 复发不乘遗忘（错题
    #    信号不是时间信号），保持 max(0.7, risk)（复发必提，见模块头）。
    def _need_gap(m: dict) -> float:
        return m.get("risk", 0) * m.get("p_forget", 1.0)

    p_need = 0.0
    evidence = None
    reason = ""
    if prereq_gaps:
        top_tid, top = max(prereq_gaps, key=lambda x: _need_gap(x[1]))
        p_need, evidence, reason = _need_gap(top), top_tid, "prereq_gap"
    if is_recurrence:
        rec = max(
            RECURRENCE_PNEED_FLOOR,
            mastery_map.get(matched_topic, {}).get("risk", 0),
        )
        if rec > p_need:
            p_need, evidence, reason = rec, matched_topic, "recurrence"
    # 教师标注的任务风险因子（safety_weight，默认 1.0）乘进 p_need：教师端 PATCH
    # 接口由此真正生效（此前只存不读是假开关）。0=永不因它预警；>1 放大过闸概率。
    # 钳 [0,1]：p_need 是概率语义，τ 恒 <1，乘超后钳 1 即必过闸。
    if evidence is not None:
        sw = mastery_map.get(evidence, {}).get("safety_weight", 1.0)
        try:
            sw = max(0.0, float(sw))
        except (TypeError, ValueError):
            sw = 1.0
        p_need = min(1.0, p_need * sw)

    # 5. cooldown：证据主题 24h 内实际拼过 → 直接沉默（防过载；AIED N=81 依据）。
    #    时间戳读 mastery_map[evidence]["last_stitched_at"]（行上自带）。
    if evidence is not None:
        ls = mastery_map.get(evidence, {}).get("last_stitched_at")
        if ls is not None:
            now_t = now if now is not None else time.time()
            if (now_t - (ls or 0)) < COOLDOWN_HOURS * 3600:
                return StitchBrief(False, None, "cooldown", p_need, 0.0)

    # 6. 单阈值两档：p_need >= τ 过闸输出完整简报。C_FA=1/C_FN=3 的代数正解是
    #    0.434；「约六天未练」语义锚点由 test_stitch_gate 实算夹死（b=0 口径）。
    tau = c_fa / (c_fa + p_need * c_fn)
    should = p_need >= tau
    return StitchBrief(
        should, evidence if should else None,
        reason if should else "below_tau", p_need, tau,
    )


async def stitch_for_turn(
    query: str, user_id: str, course_id: str, db, *, now: float | None = None
) -> StitchBrief:
    """运行时包装：算 S_t + 读 mastery/error + 调 decide_stitch。

    embed_model 在内部解析（get_embed_model 单例，失败 fail-safe 不拼），调用方无需关心。
    matched_topic 经 embedding 最近邻；无 embedding（course_topic 未灌 embedding 或不可用）
    → fail-safe 返回不拼，不阻塞回答。

    now: 时钟注入口径（评测用冻结钟保证可复现），None=各读侧自己取 time.time()
    （生产默认）。遗忘（age_days）、30 天复发窗、冷却比较统一走同一戳。
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
    hit = await _match_topic(query, course_id, db, embed_model)
    if hit is None:
        return StitchBrief(False, None, "unknown", 0.0, 1.0)
    matched_topic = hit[0]  # sim 现无消费方（confidence 因子已删），见 _match_topic

    # 2. 前置闭包 + label→topic_id 映射（各一次查询，下游复用免 N+1）
    closure = await get_prereq_predecessors(matched_topic, course_id, db)
    topic_map = await get_course_topic_map(course_id, db)

    # 3/4. mastery + error repeated（topic_map 本地对齐，无逐行 DB 查询）
    mastery_map = await _read_mastery(user_id, course_id, db, topic_map, now=now)
    error_repeated = await _read_error_repeated(user_id, course_id, db, topic_map, now=now)

    # 5. 决策（口径见 decide_stitch docstring）
    brief = decide_stitch(matched_topic, closure, mastery_map, error_repeated, now=now)

    # 6. 应拼则填简报文本（frozen dataclass 用 replace，免手工抄字段）。
    #    cooldown 戳不在此写——拼上下文发生在模型调用前，模型失败会白烧 24h 冷却；
    #    调用方（pipeline_common）把 (user, course, evidence_topic) 塞进 ctx.metadata，
    #    回合成功后由 turn_runtime 落盘（见 commit_stitch_cooldown）。
    #    matched_topic 一并带出（第一路主题过滤用，silent 时也带）。
    if brief.should_stitch and brief.evidence_topic:
        ev = mastery_map.get(brief.evidence_topic, {})
        label = ev.get("label") or brief.evidence_topic
        text = (
            f"## 学情提示（跨会话，未问亦带）\n该生在「{label}」偏薄弱"
            f"（{ev.get('observation_count', '?')}次观测），本轮涉及其前置/相关内容，请留意诊断。"
        )
        return replace(brief, text=text, matched_topic=matched_topic)
    return replace(brief, matched_topic=matched_topic)


async def commit_stitch_cooldown(db, user_id: str, course_id: str, topic_id: str) -> None:
    """回合成功后落盘 cooldown 戳（不依赖 mastery 行是否已存在）。

    旧写入端在拼上下文时就回写 row_id——两个洞：①模型调用失败也烧掉 24h 冷却；
    ②C2 复发证据来自观测日志，若该 (user, course, topic) 还没有 mastery 行则没有
    row_id，冷却不写、每轮重复拼。本函数按 (user, course, kp_id) upsert：无行时
    建占位行（theta=0/n_obs=0，观测折叠时会被覆盖），写 last_stitched_at。
    失败非致命——冷却丢一次只导致多拼一轮简报。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import select, update

    now = time.time()
    try:
        row = (
            (
                await db.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == user_id,
                        KnowledgeMastery.course_id == course_id,
                        KnowledgeMastery.kp_id == topic_id,
                    )
                )
            )
            .scalars()
            .first()
        )
        if row is None:
            # 占位行：uq (user, course, kp_id) 并发冲突时退回 UPDATE
            try:
                db.add(
                    KnowledgeMastery(
                        user_id=user_id, course_id=course_id, kp_id=topic_id,
                        label=topic_id, last_stitched_at=now,
                    )
                )
                await db.commit()
                return
            except Exception:
                await db.rollback()
        result = await db.execute(
            update(KnowledgeMastery)
            .where(
                KnowledgeMastery.user_id == user_id,
                KnowledgeMastery.course_id == course_id,
                KnowledgeMastery.kp_id == topic_id,
            )
            .values(last_stitched_at=now)
        )
        if result.rowcount != 1:
            logger.warning(
                "[stitch] cooldown 落盘命中 %d 行（期望 1）user=%s course=%s topic=%s",
                result.rowcount, user_id, course_id, topic_id,
            )
        await db.commit()
    except Exception as exc:
        logger.warning("[stitch] cooldown 落盘失败（非致命）：%s", exc)
        await db.rollback()


async def _mark_stitched(db, row_id: str) -> None:
    """（已弃用路径）按行主键回写 last_stitched_at。保留供旧调用方/评测脚本。

    生产写入口已改为 commit_stitch_cooldown（回合成功后按 (user, course, kp_id)
    upsert）。按行主键更新（非 kp_id==topic_id）——legacy 行 kp_id 是旧 hash，
    按 topic_id 条件 UPDATE 命中 0 行，last_stitched_at 永远为空，冷却失效。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import update

    try:
        result = await db.execute(
            update(KnowledgeMastery)
            .where(KnowledgeMastery.id == row_id)
            .values(last_stitched_at=time.time())
        )
        if result.rowcount != 1:
            logger.warning(
                "[stitch] last_stitched_at 回写命中 %d 行（期望 1）row_id=%s",
                result.rowcount, row_id,
            )
        await db.commit()
    except Exception as exc:
        logger.warning("[stitch] 回写 last_stitched_at 失败（非致命）：%s", exc)
        await db.rollback()


async def _match_topic(query: str, course_id: str, db, embed_model) -> tuple[str, float] | None:
    """S_t：query embedding vs course_topic.embedding 余弦最近邻。

    返回 (topic_id, sim)——调用方现不再用 sim（confidence 因子已随 VoI 三档删除），
    保留二元组签名避免无单测覆盖的签名变动；低于阈值 = unknown = None。
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
    user_id: str, course_id: str, db, topic_map: dict[str, str],
    *, now: float | None = None,
) -> dict[str, dict]:
    """读 knowledge_mastery → {topic_id: {risk, p_forget, logit, label, ...}}。

    id-align：行 kp_id 非 topic_id 时，用 label 经 topic_map（label_norm→topic_id）本地兜底，
    避免 N+1（topic_map 由 stitch_for_turn 一次取全传入）。

    掌握度估计器重构（阶段D，难度口径见 plan「班级难度估计修复」）：
    - risk 由折叠状态算出（mastery = sigmoid(theta - b_topic)，risk = 1 - mastery），
      b_topic 读 course_topic.difficulty_b（全班共享，题目难度即班级先验）；
    - k=3 班级收缩已删（与 U(n) 和 b_topic 先验功能重复且更粗），risk 即个人折叠值；
    - theta 一并放进 map——C1 入闸判据在 theta 空间（WEAK_MARGIN），班级难度 b
      不决定谁能入闸；logit = theta - b 仍带出供 risk 派生与排查；
    - 遗忘口径（MemoryBank）不变：p_forget = 1 - exp(-t/S)。
    """
    from core.db.database import CourseTopic, KnowledgeMastery
    from core.memory.course_topic_store import align_mastery_rows
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
    # id-align：单一真相源在 course_topic_store.align_mastery_rows（学生端着色共调）
    aligned = align_mastery_rows(topic_map, rows)

    # 题目难度 + safety_weight（全班共享）按**对齐后**的 tid 取齐（旧 bug：tids_hint 用
    # 原始 kp_id，legacy 哈希 id 查不到 → b=0，班级难度对旧行失效）。行内不存
    # （单一真相源在 course_topic）。
    b_map: dict[str, float] = {}
    sw_map: dict[str, float] = {}
    tids_aligned = set(aligned)
    if tids_aligned:
        b_rows = (
            await db.execute(
                select(
                    CourseTopic.topic_id,
                    CourseTopic.difficulty_b,
                    CourseTopic.safety_weight,
                ).where(
                    CourseTopic.course_id == course_id,
                    CourseTopic.topic_id.in_(tids_aligned),
                )
            )
        ).all()
        b_map = {t: float(b or 0.0) for t, b, _sw in b_rows}
        sw_map = {t: float(sw if sw is not None else 1.0) for t, _b, sw in b_rows}

    now = time.time() if now is None else now
    out: dict[str, dict] = {}
    for tid, r in aligned.items():
        b = b_map.get(tid, 0.0)
        # risk 按折叠状态重算（不读旧列——旧行可能是 delta 累加口径的历史值）；
        # theta 为 0 且无观测的行（历史遗留）等于无信息先验 risk=0.5
        logit = (r.theta or 0.0) - b
        risk = 1.0 - sigmoid(logit)
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        out[tid] = {
            # 行主键：legacy 行 kp_id 是旧 hash（靠 label 对齐进 map），_mark_stitched
            # 按 kp_id==topic_id 更新会 0 命中 → 冷却永不失效（每轮重复拼）。回写走主键。
            "row_id": r.id,
            "risk": round(risk, 4),
            "theta": round(r.theta or 0.0, 4),
            "logit": round(logit, 4),
            "p_forget": forget_prob(age_days, r.strength or 5.0),
            "strength": r.strength or 5.0,
            "label": r.label,
            "observation_count": r.observation_count or 1,
            "last_stitched_at": r.last_stitched_at,
            # 教师标注任务风险因子（默认 1.0），decide_stitch 乘进 p_need（教师 PATCH 真生效）
            "safety_weight": sw_map.get(tid, 1.0),
        }
    return out


async def _read_error_repeated(
    user_id: str, course_id: str, db, topic_map: dict[str, str],
    *, now: float | None = None,
) -> set[str]:
    """C2 复发：观测日志中同主题 poor 计数 ≥ 1（30 天窗）派生（plan §3f）。

    不再读 users.error_graph——节点 id 是对 LLM 自由标签哈希的，同一错误换个说法
    就是新节点，error_count 永远 1；且门控读仪表盘数据违反数据层约束。观测日志的
    topic_id 是闭集固定 id，数得准；source 不过滤（聊天判 poor + 答错都算复发，
    跨证据类型统一）。error_graph 退回纯仪表盘用途。
    """
    from core.db.database import MasteryObservation
    from core.memory.mastery_state import POOR_OUTCOME_MAX
    from sqlalchemy import func, select

    cutoff = (time.time() if now is None else now) - RECURRENCE_WINDOW_DAYS * 86400
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
            .having(func.count() >= 1)
        )
    ).all()
    valid = set(topic_map.values())
    return {tid for tid, _cnt in rows if tid in valid}


__all__ = [
    "StitchBrief",
    "decide_stitch",
    "stitch_for_turn",
    "commit_stitch_cooldown",
]
