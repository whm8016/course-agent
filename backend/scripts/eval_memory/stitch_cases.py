r"""拼接门控 When 评测集（evalset）：电路实验多会话正负例。

为学情拼接门控 ``decide_stitch`` 提供「输入 + 期望输出」测试用例，零 LLM、可入 CI。
正例（该拼）：前置缺口高险 / 错题复发；负例（不该拼）：unknown / 与当前提问无关 /
已纠正 / 全低险 / 阶段错位。

数据自包含：课程主题与先修关系为基于《电路分析基础实验》课程内容的人工领域标注
（领域常识，非自动抽取），用作门控回溯前置缺口的坐标。门控实现
（``core.memory.proactive.decide_stitch``）应使全部 case 的 ``gold.stitch`` 判定通过——
这是 TDD：先有测试，后有门控。

gold 核验依据（门控算法，详见 plan）：
  前置闭包 P = prereq_predecessors(matched_topic)（沿 PREREQ_EDGES 回溯）
  未问也带的候选 C1 = (P \ {matched}) ∩ {risk >= 0.5 的掌握度}
  问到了但提旧错 C2 = {matched} ∩ error_repeated
  p_need 由 C1∪C2 上的风险/复发算；过 τ 才拼；matched=None(unknown) 直接不拼。

A/无学情 vs B/无脑拼 的端到端回答质量对照（三臂评测的 baseline）由独立 runner 跑，
本文件仅提供决策层的 When 用例。
"""
from __future__ import annotations

# ---- 课程主题（topic_id → label/definition），电路分析基础实验 ----
COURSE_TOPICS: list[dict] = [
    {"topic_id": "ohm", "label": "欧姆定律", "definition": "线性电阻两端电压与电流成正比"},
    {"topic_id": "kcl", "label": "基尔霍夫电流定律", "definition": "节点电流代数和为零"},
    {"topic_id": "kvl", "label": "基尔霍夫电压定律", "definition": "回路电压代数和为零"},
    {"topic_id": "superposition", "label": "叠加定理", "definition": "线性电路多源响应可叠加"},
    {"topic_id": "thevenin", "label": "戴维南定理", "definition": "线性二端网络可等效为电压源串电阻"},
    {"topic_id": "norton", "label": "诺顿定理", "definition": "线性二端网络可等效为电流源并电阻"},
    {"topic_id": "first_order", "label": "一阶电路暂态响应", "definition": "RC/RL电路的充放电过渡过程"},
]

# ---- 先修边（src 是 dst 的直接前置）。全班共享，人工领域标注 ----
# 读法：学 dst 之前应先掌握 src。
PREREQ_EDGES: list[tuple[str, str]] = [
    ("ohm", "kcl"), ("ohm", "kvl"),
    ("kcl", "superposition"), ("kvl", "superposition"),
    ("kcl", "thevenin"), ("kvl", "thevenin"),
    ("thevenin", "norton"),
    ("thevenin", "first_order"),
]


def prereq_predecessors(topic_id: str) -> set[str]:
    """沿 PREREQ_EDGES 递归回溯 topic_id 的全部（直接+间接）前置主题。

    纯函数、可单测；门控与评测共用，保证「前置闭包」口径一致。
    """
    preds: set[str] = set()
    frontier = [topic_id]
    while frontier:
        cur = frontier.pop()
        for src, dst in PREREQ_EDGES:
            if dst == cur and src not in preds:
                preds.add(src)
                frontier.append(src)
    return preds


# ---- 学生 case：掌握度状态 + 错题复发 + 本轮提问 + 提问命中主题 + gold ----
# mastery:        该生在各主题的掌握度（risk >= 0.5 视为薄弱）
# error_repeated: 反复出错的主题 id 列表（matched 命中其一即复发信号 C2）
# query:          本轮学生提问
# matched_topic:  提问经 S_t embedding 最近邻命中的主题（None = unknown 拒答）
# gold.stitch:    门控应否把学情简报拼入本轮回答
# gold.reason:    触发/不触发的原因标签
# gold.evidence_topic: 该拼时，简报应指向的证据主题
STITCH_CASES: list[dict] = [
    # ===== 正例：该拼 =====
    {
        "id": "pos_prereq_basic",
        "label": "前置缺口：欧姆定律薄弱却问KCL",
        "mastery": [{"topic_id": "ohm", "risk": 0.85, "mastery": 0.2, "observation_count": 4}],
        "error_repeated": [],
        "query": "节点电流方程怎么列",
        "matched_topic": "kcl",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "ohm"},
    },
    {
        "id": "pos_prereq_deep",
        "label": "深层前置缺口：KCL薄弱却问戴维南",
        "mastery": [{"topic_id": "kcl", "risk": 0.8, "mastery": 0.25, "observation_count": 3}],
        "error_repeated": [],
        "query": "怎么求戴维南等效电阻",
        "matched_topic": "thevenin",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "kcl"},
    },
    {
        "id": "pos_recurrence",
        "label": "错题复发：戴维南反复错又问戴维南",
        "mastery": [{"topic_id": "thevenin", "risk": 0.75, "mastery": 0.3, "observation_count": 5}],
        "error_repeated": ["thevenin"],
        "query": "戴维南等效电路怎么画",
        "matched_topic": "thevenin",
        "gold": {"stitch": True, "reason": "recurrence", "evidence_topic": "thevenin"},
    },
    {
        "id": "pos_prereq_recurrence",
        "label": "前置缺口叠加复发：欧姆反复错问KCL",
        "mastery": [{"topic_id": "ohm", "risk": 0.9, "mastery": 0.15, "observation_count": 6}],
        "error_repeated": ["ohm"],
        "query": "KCL和KVL有什么区别",
        "matched_topic": "kcl",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "ohm"},
    },
    # ===== 负例：不该拼 =====
    {
        "id": "neg_unknown",
        "label": "未知问题：不命中任何主题",
        "mastery": [{"topic_id": "ohm", "risk": 0.85, "mastery": 0.2, "observation_count": 4}],
        "error_repeated": [],
        "query": "实验报告下周五几点交",
        "matched_topic": None,
        "gold": {"stitch": False, "reason": "unknown"},
    },
    {
        "id": "neg_irrelevant",
        "label": "无关：暂态薄弱却问欧姆定律",
        "mastery": [{"topic_id": "first_order", "risk": 0.85, "mastery": 0.2, "observation_count": 4}],
        "error_repeated": [],
        "query": "欧姆定律的公式是什么",
        "matched_topic": "ohm",
        "gold": {"stitch": False, "reason": "irrelevant"},
    },
    {
        "id": "neg_corrected",
        "label": "已纠正：欧姆曾错已纠正，问KCL不再缺口",
        "mastery": [{"topic_id": "ohm", "risk": 0.2, "mastery": 0.85, "observation_count": 6}],
        "error_repeated": [],
        "query": "节点电流方程怎么列",
        "matched_topic": "kcl",
        "gold": {"stitch": False, "reason": "corrected"},
    },
    {
        "id": "neg_all_low",
        "label": "全低险：无薄弱点",
        "mastery": [
            {"topic_id": "ohm", "risk": 0.3, "mastery": 0.8, "observation_count": 3},
            {"topic_id": "kcl", "risk": 0.35, "mastery": 0.75, "observation_count": 3},
        ],
        "error_repeated": [],
        "query": "怎么求戴维南等效电阻",
        "matched_topic": "thevenin",
        "gold": {"stitch": False, "reason": "no_weak"},
    },
    {
        "id": "neg_stage_mismatch",
        "label": "阶段错位：诺顿薄弱却问基础欧姆",
        "mastery": [{"topic_id": "norton", "risk": 0.85, "mastery": 0.2, "observation_count": 4}],
        "error_repeated": [],
        "query": "欧姆定律怎么用",
        "matched_topic": "ohm",
        "gold": {"stitch": False, "reason": "stage_mismatch"},
    },
]
