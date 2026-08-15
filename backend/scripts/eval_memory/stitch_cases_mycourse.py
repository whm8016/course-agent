r"""mycourse 真实场景拼接门控评测集（``course_topic`` / ``course_topic_edge`` 离线抽取图）。

与 ``stitch_cases``（7 主题人工 seed 图）不同，本文件用 **mycourse 真实 topic_id**（与
``data/course_topics/course_mycourse_topics.json`` / DB 一致），gold 经
``get_prereq_predecessors`` + ``decide_stitch`` 对真实前置边验过。

前置闭包在 L1 scorer 与 L2 e2e 均从 DB 读（``course_topic_store.get_prereq_predecessors``），
不再用内存 ``PREREQ_EDGES``。
"""
from __future__ import annotations

COURSE_ID = "mycourse"

# topic_id 与 DB / 导出 JSON 一致（label 归一化截断 32 字符）
STITCH_CASES: list[dict] = [
    # ===== 正例：该拼 =====
    {
        "id": "pos_prereq_equiv_r",
        "label": "前置缺口：线性电阻薄弱却问等效内阻测量",
        "mastery": [
            {"topic_id": "线性电阻元件的伏安特性", "risk": 0.85, "mastery": 0.2, "observation_count": 4},
        ],
        "error_repeated": [],
        "query": "等效内阻怎么测量？开路短路法具体步骤是什么？",
        "matched_topic": "等效内阻测量",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "线性电阻元件的伏安特性"},
    },
    {
        "id": "pos_prereq_thevenin",
        "label": "前置缺口：实际电压源薄弱却问戴维南定理",
        "mastery": [
            {"topic_id": "实际电压源的伏安特性", "risk": 0.8, "mastery": 0.25, "observation_count": 3},
        ],
        "error_repeated": [],
        "query": "戴维南定理怎么用？怎么求等效电压和等效电阻？",
        "matched_topic": "戴维南定理",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "实际电压源的伏安特性"},
    },
    {
        "id": "pos_prereq_potentiometer",
        "label": "前置缺口：三端变阻器薄弱却问分压器调压",
        "mastery": [
            {"topic_id": "三端变阻器的连接方式", "risk": 0.85, "mastery": 0.2, "observation_count": 4},
        ],
        "error_repeated": [],
        "query": "分压器怎么调压？滑动端应该接在哪里？",
        "matched_topic": "分压器调压特性",
        "gold": {"stitch": True, "reason": "prereq_gap", "evidence_topic": "三端变阻器的连接方式"},
    },
    {
        "id": "pos_recurrence_thevenin",
        "label": "错题复发：戴维南反复错又问外特性验证",
        "mastery": [
            {"topic_id": "戴维南定理", "risk": 0.75, "mastery": 0.3, "observation_count": 5},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南等效电路的外特性怎么验证？",
        "matched_topic": "戴维南等效电路外特性验证",
        "gold": {"stitch": True, "reason": "recurrence", "evidence_topic": "戴维南定理"},
    },
    # ===== 负例：不该拼 =====
    {
        "id": "neg_unknown",
        "label": "未知问题：不命中任何课程主题",
        "mastery": [
            {"topic_id": "线性电阻元件的伏安特性", "risk": 0.85, "mastery": 0.2, "observation_count": 4},
        ],
        "error_repeated": [],
        "query": "实验报告下周五几点交到教务处？",
        "matched_topic": None,
        "gold": {"stitch": False, "reason": "unknown"},
    },
    {
        "id": "neg_irrelevant",
        "label": "无关：滤波器薄弱却问线性电阻伏安特性",
        "mastery": [
            {"topic_id": "滤波器设计", "risk": 0.85, "mastery": 0.2, "observation_count": 4},
        ],
        "error_repeated": [],
        "query": "线性电阻的伏安特性怎么测量？",
        "matched_topic": "线性电阻元件的伏安特性",
        "gold": {"stitch": False, "reason": "irrelevant"},
    },
    {
        "id": "neg_corrected",
        "label": "已纠正：线性电阻曾错已纠正，问等效内阻不再缺口",
        "mastery": [
            {"topic_id": "线性电阻元件的伏安特性", "risk": 0.2, "mastery": 0.85, "observation_count": 6},
        ],
        "error_repeated": [],
        "query": "等效内阻测量时为什么要先测开路电压？",
        "matched_topic": "等效内阻测量",
        "gold": {"stitch": False, "reason": "corrected"},
    },
    {
        "id": "neg_all_low",
        "label": "全低险：前置主题均已掌握",
        "mastery": [
            {"topic_id": "线性电阻元件的伏安特性", "risk": 0.3, "mastery": 0.8, "observation_count": 3},
            {"topic_id": "实际电压源的伏安特性", "risk": 0.35, "mastery": 0.75, "observation_count": 3},
        ],
        "error_repeated": [],
        "query": "戴维南定理和诺顿定理有什么区别？",
        "matched_topic": "戴维南定理",
        "gold": {"stitch": False, "reason": "no_weak"},
    },
    {
        "id": "neg_stage_mismatch",
        "label": "阶段错位：分压变阻器选值薄弱却问基础伏安特性",
        "mastery": [
            {"topic_id": "分压变阻器阻值与容量选取", "risk": 0.85, "mastery": 0.2, "observation_count": 4},
        ],
        "error_repeated": [],
        "query": "线性电阻元件的伏安特性曲线怎么画？",
        "matched_topic": "线性电阻元件的伏安特性",
        "gold": {"stitch": False, "reason": "stage_mismatch"},
    },
]
