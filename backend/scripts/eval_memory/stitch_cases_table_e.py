r"""表E评测集：门控决策 vs 教师规则金标（mycourse 课文锚定图 v2，30 例）。

评测对象是 ``core.memory.proactive.decide_stitch`` 的**决策层**（该不该把未问及的
跨会话学情拼进本轮回答），不含 S_t 问句→主题 embedding 匹配（matched_topic 是档案
给定值）。图源为 ``graph_mycourse_v2.json`` 快照（193 主题 / 225 边，与 DB 同源导出，
零 DB 依赖）。

与 ``stitch_cases``（TDD 用例，金标=门控算法推演，全对是必然）的本质区别：
**金标=教师规则句子**（``teacher_rule``，教师能点头/摇头的话），不是算法公式。门控
是这些句子的数值实现（薄弱折叠 × 遗忘概率 vs 阈值），两者不是一回事——门控可能因
冷却、遗忘耦合、班级难度而偏离句子，**偏离处正是评测的信息量**。预期门控在冷却类
（cooldown_1h / cooldown_20h）漏提醒：金标按「连问两次说明真不会」定该提，门控按
防打扰设计沉默——这是真实的设计争议，有失败分析的评测比 9/9 满分可信。

档案构造（真实折叠，非手拍数值）：
- ``levels``：闭集判定等级序列（poor/fair/good），经 ``fold_observations``（Elo 折叠
  W_JUDGE=0.6）算 theta/risk——teacher_rule 里的「两次判未掌握」直接映射序列长度。
- ``days_since_last``：距最后一次判定观测的天数，经 ``forget_prob``（S=5）算 p_forget
  ——「昨天刚练过」「快一个月没练」映射遗忘概率。
- ``last_stitched_hours_ago``：同证据主题上次实际拼入简报的小时数（冷却判定），
  锚定 **决策钟** case_now——冷却是「本轮前 N 小时」的运行时概念，与种钟解耦。
- ``b_topic``：班级难度（仅 neg_class_hard，题目难度不决定谁能入闸的回归锚）。
- ``error_repeated``：近 30 天同主题 poor ≥2 的复发证据（C2 候选口径）；观测间距
  由它定——在列 1 天距（≥2 poor 自然落进 30 天窗）、不在 32 天距（窗内至多 1 条），
  e2e 落库后复发由读侧自己数出、自检对账。
- 时钟：观测 created_at 锚定**种钟**（模块 NOW）；case 可选 ``now`` 键只挪**决策钟**
  （七、时钟三态：同档案不同今天），遗忘/复发窗/冷却全由管道按 now 实算。

teacher 权重（safety_weight）用例不收：教师不会真去标（用户决策 2026-08-16），
金标句子不建立在教师标注场景上。

金标方向（用户逐句点头，2026-08-16）：
- 刚练过（1~5 天）的薄弱先修=不提（遗忘耦合被教师规则背书，与「单次失手别小题大做」同口径）；
- 冷却内连问=该提（门控漏，设计争议点）；
- 补一半（poor,poor,good）=不提（净欠账≈单次失手，先观望）；
- 仪器兄弟主题（同仪器不同功能，互非先修）=不提。
"""
from __future__ import annotations

import json
from pathlib import Path

# 固定时钟：冷却/遗忘判定确定性可测（decide_stitch 的 now 参数）
NOW = 1_800_000_000.0

_SNAPSHOT = Path(__file__).resolve().parent / "graph_mycourse_v2.json"


def load_graph() -> dict:
    """快照图（确定性）：{topics: [{topic_id,label}], edges: [{src,dst,relation}]}。"""
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def prereq_predecessors(topic_id: str, edges: list[dict]) -> set[str]:
    """快照边 → 先修闭包（复用 ``course_topic_store.prereq_closure``，关系过滤
    ``PREREQ_RELATIONS``——口径的单一陈述见该函数 docstring）。"""
    from core.memory.course_topic_store import PREREQ_RELATIONS, prereq_closure

    return prereq_closure(
        topic_id,
        ((e["src"], e["dst"]) for e in edges if e["relation"] in PREREQ_RELATIONS),
    )


def case_now(case: dict) -> float:
    """case 决定时钟：默认模块 NOW；「时钟三态」例显式覆盖（同档案不同今天）。"""
    return case.get("now", NOW)


def entry_observations(case: dict, m: dict, *, base: float | None = None) -> list[dict]:
    """mastery 条目 → 观测日志（离线折叠与 e2e 落库共用同一份构造源）。

    时钟约定：观测锚定**种钟**（默认模块 NOW；base 显式覆盖——live 层用运行时
    墙钟作锚，间距/天数推导仍走这一份单一真相源）——levels[-1] = t_last = 种钟 -
    days_since_last*86400（最后接触时刻），更早的档按间距往回倒；fold_observations
    入口自按 created_at 升序排（Elo 顺序相关），相对序不变则折出的 theta 逐位不变。
    间距按复发口径定：条目在 case.error_repeated → 1 天（本主题 poor 落进 30 天窗，
    读侧按 ≥1 自然数出）；不在 → 窗宽+2=32 天（旧 poor 推出窗外——≥1 口径下该主题
    窗内无 poor，不制造复发）。复发计数门槛 2026-08-18 由 ≥2 下调到 ≥1（与
    WEAK_MARGIN「一次薄弱即入闸」同语义）。
    """
    from core.memory.mastery_state import LEVEL_OUTCOME, W_JUDGE
    from core.memory.proactive import RECURRENCE_WINDOW_DAYS

    spacing = (
        86400.0
        if m["topic_id"] in case.get("error_repeated", [])
        else (RECURRENCE_WINDOW_DAYS + 2) * 86400.0
    )
    t_last = (NOW if base is None else base) - m["days_since_last"] * 86400
    n = len(m["levels"])
    return [
        {
            "created_at": t_last - (n - 1 - j) * spacing,
            "outcome": LEVEL_OUTCOME[lv],
            "weight": W_JUDGE,
        }
        for j, lv in enumerate(m["levels"])
    ]


def entry_last_stitched_at(case: dict, m: dict) -> float | None:
    """冷却戳锚定 case 自己的**决策钟**（「本轮前 N 小时」的运行时概念，与种钟
    解耦——三态例只挪 now 也能正确演冷却）。"""
    if "last_stitched_hours_ago" not in m:
        return None
    return case_now(case) - m["last_stitched_hours_ago"] * 3600


def gate_decide(case: dict, edges: list[dict]):
    """case → 门控决策一条龙（闭包 + 档案构建 + decide_stitch）。

    runner 与测试共用同一管线——评测测的与跑的必然是同一个决策路径。
    决策钟走 case_now（默认 NOW；三态例显式覆盖）。
    """
    from core.memory.proactive import decide_stitch

    matched = case["matched_topic"]
    closure = prereq_predecessors(matched, edges) if matched else set()
    return decide_stitch(
        matched, closure, build_mastery_map(case), set(case["error_repeated"]),
        now=case_now(case),
    )


def build_mastery_map(case: dict) -> dict[str, dict]:
    """case 档案 → decide_stitch 的 mastery_map（真实折叠 + 遗忘实算）。

    观测/冷却戳走 entry_observations/entry_last_stitched_at（与 e2e 落库同一
    构造源；e2e 侧 theta 由生产写侧 consolidation._write_mastery_observations
    从库内观测全量重折，这里离线直折同一份 obs，两侧同 regime）。p_forget 按
    「决策钟 - 最后观测时刻」实算（同 _read_mastery 的 age 口径），对无 now 键的
    旧例与 days_since_last 直传逐位一致。b≠0 时从 b 起折（consolidation 同款
    init，非 b=0 折完读时补减）；b 只进 risk/logit 不进入闸判据（theta 空间）。
    字段集对齐 _read_mastery / scorer._gate_mastery_map 的 map 形状（label 便于
    调试）；decide_stitch 只读 risk/theta/p_forget/last_stitched_at（.get 默认）。
    """
    from core.memory.mastery import forget_prob
    from core.memory.mastery_state import FoldState, fold_observations

    out: dict[str, dict] = {}
    for m in case["mastery"]:
        obs = entry_observations(case, m)
        s = fold_observations(obs, FoldState(b_topic=m.get("b_topic", 0.0)))
        age_days = (case_now(case) - max(o["created_at"] for o in obs)) / 86400.0
        out[m["topic_id"]] = {
            "risk": round(s.risk, 4),
            "theta": round(s.theta, 4),
            "logit": round(s.logit, 4),
            "p_forget": forget_prob(age_days),
            "observation_count": s.n_obs,
            "label": m.get("label", m["topic_id"]),
            "last_stitched_at": entry_last_stitched_at(case, m),
        }
    return out


# ---- 27 例：档案 + 教师规则金标（句子经用户逐条点头，2026-08-16）----
# gold.stitch: 教师规则下的「该不该把未问及的学情拼进本轮回答」
# gold.evidence_topic: 该提时简报应指向的主题（次级指标，不进主指标）
STITCH_CASES: list[dict] = [
    # ===== 一、前置缺口（该提）=====
    {
        "id": "pos_gap_ohm_superposition",
        "label": "前置缺口：欧姆定律薄弱问叠加定理",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "叠加要拆成单源单独作用，每一步都在用欧姆；地基两次没掌握又快一个月没练，该提醒",
        "gold": {"stitch": True, "evidence_topic": "欧姆定律"},
    },
    {
        "id": "pos_gap_thevenin_equiv",
        "label": "前置缺口：戴维南定理薄弱问等效电路",
        "mastery": [
            {"topic_id": "戴维南定理", "label": "戴维南定理", "levels": ["poor", "poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南等效电路怎么求？等效电压和电阻分别怎么算？",
        "matched_topic": "戴维南等效电路",
        "teacher_rule": "这正是要用定理的地方，该提醒",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "pos_gap_potentiometer",
        "label": "前置缺口：三端变阻器薄弱问分压调压",
        "mastery": [
            {"topic_id": "三端变阻器的应用", "label": "三端变阻器的应用", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["三端变阻器的应用"],
        "query": "分压器的调压特性实验怎么做？滑动端接在哪里？",
        "matched_topic": "分压器的调压特性",
        "teacher_rule": "接法不对调压根本做不下去，该提醒",
        "gold": {"stitch": True, "evidence_topic": "三端变阻器的应用"},
    },
    {
        "id": "pos_gap_wien_amp",
        "label": "前置缺口：文氏电路薄弱问幅频特性",
        "mastery": [
            {"topic_id": "文氏电路", "label": "文氏电路", "levels": ["poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["文氏电路"],
        "query": "文氏电路的幅频特性怎么测？谐振点怎么找？",
        "matched_topic": "文氏电路的幅频特性",
        "teacher_rule": "该提醒",
        "gold": {"stitch": True, "evidence_topic": "文氏电路"},
    },
    {
        "id": "pos_gap_deep_chain",
        "label": "深层前置缺口：含源一端口薄弱问戴维南等效电路（隔两级）",
        "mastery": [
            {"topic_id": "含源一端口网络", "label": "含源一端口网络", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["含源一端口网络"],
        "query": "戴维南等效电路的外特性怎么验证？",
        "matched_topic": "戴维南等效电路",
        "teacher_rule": "隔着一级的地基也是空的，该提醒",
        "gold": {"stitch": True, "evidence_topic": "含源一端口网络"},
    },
    {
        "id": "pos_gap_multimeter_instrument",
        "label": "仪器先修：万用表使用薄弱问电阻电容测量",
        "mastery": [
            {"topic_id": "万用表的使用", "label": "万用表的使用", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["万用表的使用"],
        "query": "数字万用表怎么测电阻和电容？要选什么档位？",
        "matched_topic": "数字万用表测量电阻电容值",
        "teacher_rule": "表都不会用，测量课寸步难行，该提醒",
        "gold": {"stitch": True, "evidence_topic": "万用表的使用"},
    },
    {
        "id": "pos_gap_rc_scope_method",
        "label": "前置缺口：rc微分电路薄弱问示波器观察过渡过程",
        "mastery": [
            {"topic_id": "rc微分电路", "label": "RC微分电路", "levels": ["poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["rc微分电路"],
        "query": "怎么用示波器观察电路的过渡过程？",
        "matched_topic": "用示波器观察电路过渡过程的方法",
        "teacher_rule": "要观察的现象正是他没掌握的电路产生的，该提醒",
        "gold": {"stitch": True, "evidence_topic": "rc微分电路"},
    },
    # ===== 二、复发（该提）=====
    {
        "id": "pos_recurrence_thevenin",
        "label": "复发：戴维南反复错又问戴维南",
        "mastery": [
            {"topic_id": "戴维南定理", "label": "戴维南定理", "levels": ["poor", "poor"], "days_since_last": 3},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南定理的等效电阻还是没算对，再讲一遍？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "旧错就在眼前，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "pos_recurrence_rc",
        "label": "复发：rc微分电路反复错又问它",
        "mastery": [
            {"topic_id": "rc微分电路", "label": "RC微分电路", "levels": ["poor", "poor"], "days_since_last": 2},
        ],
        "error_repeated": ["rc微分电路"],
        "query": "rc微分电路的输出波形为什么是尖脉冲？",
        "matched_topic": "rc微分电路",
        "teacher_rule": "反复错又问它，该提",
        "gold": {"stitch": True, "evidence_topic": "rc微分电路"},
    },
    {
        "id": "pos_recurrence_beats_gap",
        "label": "复发与前置缺口竞争：先提反复失手的",
        "mastery": [
            {"topic_id": "叠加定理", "label": "叠加定理", "levels": ["poor", "poor"], "days_since_last": 2},
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 30},
        ],
        "error_repeated": ["叠加定理", "欧姆定律"],
        "query": "叠加定理拆电源的时候功率为什么不叠加？",
        "matched_topic": "叠加定理",
        "teacher_rule": "两头都该提时，先提反复失手的那件",
        "gold": {"stitch": True, "evidence_topic": "叠加定理"},
    },
    # ===== 三、遗忘耦合（刚练过=不提，用户已定）=====
    {
        "id": "fc_gap_seen_yesterday",
        "label": "遗忘耦合：薄弱但昨天刚练过",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "知道他薄弱，但昨天刚在练这块——先别打扰，看这轮表现再说",
        "gold": {"stitch": False},
    },
    {
        "id": "fc_gap_week_stale",
        "label": "遗忘耦合：薄弱且整整一周没练",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 7},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "两次没掌握又隔了一周，差不多该忘了，该提醒",
        "gold": {"stitch": True, "evidence_topic": "欧姆定律"},
    },
    {
        "id": "fc_gap_five_days",
        "label": "遗忘耦合：薄弱但才隔五天",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 5},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "才隔五天，还在保鲜期里，不必主动提",
        "gold": {"stitch": False},
    },
    # ===== 四、冷却（预定失败区：金标=该提，门控防打扰沉默）=====
    {
        "id": "cooldown_1h",
        "label": "冷却争议：1小时前刚提醒过又问",
        "mastery": [
            {
                "topic_id": "戴维南定理", "label": "戴维南定理",
                "levels": ["poor", "poor"], "days_since_last": 3,
                "last_stitched_hours_ago": 1,
            },
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南等效电阻还是不懂，刚才那个提醒没看明白",
        "matched_topic": "戴维南定理",
        "teacher_rule": "一小时内连问两次，说明刚才的提醒没解决问题，该再提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "cooldown_20h",
        "label": "冷却争议：20小时前提醒过又问",
        "mastery": [
            {
                "topic_id": "戴维南定理", "label": "戴维南定理",
                "levels": ["poor", "poor"], "days_since_last": 3,
                "last_stitched_hours_ago": 20,
            },
        ],
        "error_repeated": ["戴维南定理"],
        "query": "还是戴维南那里卡住了，帮我再看看",
        "matched_topic": "戴维南定理",
        "teacher_rule": "二十小时后还卡在同一处，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "cooldown_expired_25h",
        "label": "冷却过期：25小时前提醒过又问",
        "mastery": [
            {
                "topic_id": "戴维南定理", "label": "戴维南定理",
                "levels": ["poor", "poor"], "days_since_last": 3,
                "last_stitched_hours_ago": 25,
            },
        ],
        "error_repeated": ["戴维南定理"],
        "query": "隔了一天再问戴维南，等效电阻怎么求？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "隔了一整天又问，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    # ===== 五、干净负例（不该提）=====
    {
        "id": "neg_unknown",
        "label": "未知问题：不命中任何课程主题",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "实验报告下周五几点交到教务处？",
        "matched_topic": None,
        "teacher_rule": "跟课程内容无关的问题，不插学情",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_irrelevant",
        "label": "无关：并联谐振薄弱问最基础的伏安测试",
        "mastery": [
            {"topic_id": "并联谐振", "label": "并联谐振", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["并联谐振"],
        "query": "元件的伏安特性怎么测试？要接什么仪表？",
        "matched_topic": "元件的伏安特性的测试",
        "teacher_rule": "问的是最基础内容，薄弱的是不相干的章节，不提",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_corrected",
        "label": "已纠正：欧姆曾两次未掌握后连三次掌握",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor", "good", "good", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "补上了就别翻旧账",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_partial_recovery",
        "label": "补一半：两次未掌握只补回一次好表现（用户定：先观望）",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "刚有起色先观望，净欠账约等于一次失手，别小题大做",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_all_solid",
        "label": "全低险：叠加定理五个先修全判掌握",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["good", "good"], "days_since_last": 30},
            {"topic_id": "基尔霍夫电流定律", "label": "基尔霍夫电流定律", "levels": ["good", "good"], "days_since_last": 30},
            {"topic_id": "电阻串并联电路", "label": "电阻串并联电路", "levels": ["good", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "要用的先修都判了掌握，地基是牢的，不提",
        "gold": {"stitch": False},
    },
    {
        # 2026-08-18 弱线降为「一次判 poor 即入闸」后翻正（原 gold=False）：
        # 一次失手 + 三周未碰 = 该回头看一眼，不再是"别小题大做"
        "id": "neg_single_poor",
        "label": "单次失手：欧姆错过一次、三周没碰（08-18 口径翻正）",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor"], "days_since_last": 21},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "错过一次又三周没碰，该回头看一眼",
        "gold": {"stitch": True, "evidence_topic": "欧姆定律"},
    },
    {
        "id": "neg_class_hard",
        "label": "班级难度：戴维南全班难(b=0.5)但该生表现正常",
        "mastery": [
            {"topic_id": "戴维南定理", "label": "戴维南定理", "levels": ["fair", "fair", "fair"], "days_since_last": 30, "b_topic": 0.5},
        ],
        "error_repeated": [],
        "query": "戴维南等效电路怎么求？",
        "matched_topic": "戴维南等效电路",
        "teacher_rule": "题难不等于这个学生薄弱，不提",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_instrument_sibling",
        "label": "仪器兄弟：示波器光标测量薄弱问垂直系统设置",
        "mastery": [
            {"topic_id": "示波器光标测量", "label": "示波器光标测量", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["示波器光标测量"],
        "query": "示波器垂直系统怎么设置？ VOLTS/DIV 旋钮干什么用？",
        "matched_topic": "示波器垂直系统设置",
        "teacher_rule": "问哪块补哪块，同一台仪器的兄弟功能不算先修，不提",
        "gold": {"stitch": False},
    },
    # ===== 六、多缺口竞争 + 边界 =====
    {
        "id": "multi_gap_stale_wins",
        "label": "多缺口：先提快忘掉的（正练着的不急）",
        "mastery": [
            {"topic_id": "三端变阻器的应用", "label": "三端变阻器的应用", "levels": ["poor", "poor"], "days_since_last": 30},
            {"topic_id": "线性电路", "label": "线性电路", "levels": ["poor", "poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["三端变阻器的应用", "线性电路"],
        "query": "分压器的调压特性实验要注意什么？",
        "matched_topic": "分压器的调压特性",
        "teacher_rule": "两个先修都薄弱，先提快忘掉的那个，正在练的不急",
        "gold": {"stitch": True, "evidence_topic": "三端变阻器的应用"},
    },
    {
        "id": "multi_gap_weaker_wins",
        "label": "多缺口：都久没练时提更薄弱的",
        "mastery": [
            {"topic_id": "三端变阻器的应用", "label": "三端变阻器的应用", "levels": ["poor", "poor"], "days_since_last": 30},
            {"topic_id": "线性电路", "label": "线性电路", "levels": ["poor", "poor", "poor"], "days_since_last": 30},
        ],
        "error_repeated": ["三端变阻器的应用", "线性电路"],
        "query": "分压器的调压特性实验要注意什么？",
        "matched_topic": "分压器的调压特性",
        "teacher_rule": "都久没练，提更薄弱的那个",
        "gold": {"stitch": True, "evidence_topic": "线性电路"},
    },
    {
        "id": "neg_answer_is_teaching",
        "label": "回答即教学：问的就是自己薄弱的主题本身（旧账已出复发窗）",
        "mastery": [
            {"topic_id": "rc微分电路", "label": "RC微分电路", "levels": ["poor", "poor"], "days_since_last": 45},
        ],
        "error_repeated": [],
        "query": "rc微分电路的原理是什么？输出输入什么关系？",
        "matched_topic": "rc微分电路",
        "teacher_rule": "这轮回答本身就在教这块，不用再叠一层预警",
        "gold": {"stitch": False},
    },
    # ===== 七、时钟三态（2026-08-17 补：同档案不同今天，遗忘/冷却由管道按 now 实算）=====
    # 种子同一份：欧姆定律 2×poor、最后接触在 NOW-1 天、问叠加定理；三例只差
    # 「今天」是哪天 / 一小时前有没有提醒过。days_since_last 固定 1、只把 now 加
    # 27 天（day28 例年龄 = 1+27 = 28 天；两边都改成 28 会变 55 天，不许）。
    {
        "id": "clock_day1_silent",
        "label": "时钟三态：刚练过一天（保鲜期内，沉默）",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "昨天刚练过，还在保鲜期，先别打扰",
        "gold": {"stitch": False},
    },
    {
        "id": "clock_day28_stitch",
        "label": "时钟三态：同一档案四周后（生疏了，该提）",
        "mastery": [
            {"topic_id": "欧姆定律", "label": "欧姆定律", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "同一个档案，四周后还卡在叠加定理，地基早该提了",
        "gold": {"stitch": True, "evidence_topic": "欧姆定律"},
        "now": NOW + 27 * 86400,
    },
    {
        "id": "clock_day28_cooldown",
        "label": "时钟三态：四周后且一小时前刚提醒过（冷却，门控沉默是设计争议）",
        "mastery": [
            {
                "topic_id": "欧姆定律", "label": "欧姆定律",
                "levels": ["poor", "poor"], "days_since_last": 1,
                "last_stitched_hours_ago": 1,
            },
        ],
        "error_repeated": ["欧姆定律"],
        "query": "叠加定理怎么用？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理",
        "teacher_rule": "一小时前刚提醒过同一件事；门控按防打扰沉默（争议预定漏提醒，同 cooldown 两例口径）",
        "gold": {"stitch": True, "evidence_topic": "欧姆定律"},
        "now": NOW + 27 * 86400,
    },
]
