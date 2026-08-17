"""表E评测集（stitch_cases_table_e）：完整性 + 公式行为回归锁。

两类断言分工（2026-08-17 起）：完整性/闭包锚/冷却 reason/证据选择/遗忘耦合/
时钟三态是**公式行为锁**——锁决策函数的机制，不随评测结果动；混淆矩阵形状
**不再用测试钉死**（旧「偏离必须恰好冷却两例」「打扰率必须为 0」两条已删）——
金标=教师规则，门控因匹配/真难度/冷却产生的偏离是评测的信息量，由 e2e 的
match_ok / cooldown_fn 列去量，不是要修的 bug。防自证底线保留：gate 均衡
准确率 < 1.0（满分=金标照算法写的自证回退）。
"""
from __future__ import annotations

from scripts.eval_memory.run_stitch_table_e import ZERO_LLM_ARMS, run
from scripts.eval_memory.stitch_cases_table_e import (
    STITCH_CASES, gate_decide, load_graph, prereq_predecessors,
)

_EDGES = load_graph()["edges"]


def _gate_decide(case: dict):
    """与 runner 同一条决策管线（gate_decide 单一来源，测试=评测口径）。"""
    return gate_decide(case, _EDGES)


# ── 用例完整性：档案引用的主题必须都在快照图里 ─────────────────────────────


def test_case_count_and_rule_sentences():
    """30 例；每例 teacher_rule 是非空句子（教师可点头/摇头）；金标与证据自洽。"""
    assert len(STITCH_CASES) == 30
    ids = [c["id"] for c in STITCH_CASES]
    assert len(set(ids)) == len(ids), "case id 重复"
    for c in STITCH_CASES:
        assert c["teacher_rule"].strip(), f"{c['id']} 缺教师规则句子"
        assert isinstance(c["gold"]["stitch"], bool)
        if c["gold"]["stitch"]:
            assert c["gold"].get("evidence_topic"), f"{c['id']} 该提却无证据主题"
        else:
            assert not c["gold"].get("evidence_topic"), f"{c['id']} 不该提却有证据主题"


def test_cases_reference_real_graph_topics():
    """matched_topic / mastery 主题 / error_repeated / 金标证据全部落在快照图内。"""
    graph = load_graph()
    tids = {t["topic_id"] for t in graph["topics"]}
    assert len(tids) == 193, "快照应是课文锚定新图（193 主题）；图换了请重跑 export_graph_snapshot"
    for c in STITCH_CASES:
        if c["matched_topic"] is not None:
            assert c["matched_topic"] in tids, f"{c['id']} matched 不在图中"
        for m in c["mastery"]:
            assert m["topic_id"] in tids, f"{c['id']} mastery 主题 {m['topic_id']} 不在图中"
        for t in c["error_repeated"]:
            assert t in tids, f"{c['id']} error_repeated 主题 {t} 不在图中"
        ev = c["gold"].get("evidence_topic")
        if ev:
            assert ev in tids, f"{c['id']} 金标证据 {ev} 不在图中"


def test_closure_samples():
    """闭包抽验：锚点主题的真实先修关系。"""
    assert prereq_predecessors("叠加定理", _EDGES) >= {"欧姆定律", "基尔霍夫电流定律", "电阻串并联电路"}
    assert prereq_predecessors("戴维南等效电路", _EDGES) == {"含源一端口网络", "戴维南定理"}
    assert prereq_predecessors("示波器垂直系统设置", _EDGES) == {"示波器的使用"}
    assert prereq_predecessors("元件的伏安特性的测试", _EDGES) == set()  # 根主题
    assert "示波器光标测量" not in prereq_predecessors("示波器垂直系统设置", _EDGES)  # 仪器兄弟非先修


# ── 防自证：门控与教师金标必须存在真实偏离 ──────────────────────────────────


def test_gate_misses_cooldown_cases():
    """冷却争议例（cooldown_1h/20h）：金标=该提（连问说明真不会），门控防打扰沉默。

    这两条 FN 是评测的信息量所在（设计争议点），不是要修的 bug——把它们改没或
    让门控全对，等于把金标退回「照算法写」的自证老路。
    """
    for cid in ("cooldown_1h", "cooldown_20h"):
        case = next(c for c in STITCH_CASES if c["id"] == cid)
        brief = _gate_decide(case)
        assert case["gold"]["stitch"] is True
        assert brief.should_stitch is False, f"{cid} 应被冷却压住（FN 预定）"
        assert brief.reason == "cooldown"


def test_gate_not_perfect_and_baselines_anchor():
    """防自证底线：gate 均衡准确率 < 1.0；never/always 两基线恒 0.5（口径锚）。

    打扰率/偏离面不再钉死（金标=教师规则，偏离是评测的信息量）；runner 的
    run() 是 async，这里用 asyncio 直接驱动，零 LLM 三臂不碰网络。
    """
    import asyncio

    summary = asyncio.run(run(with_llm=False))
    arms = summary["arms"]
    assert set(arms) == set(ZERO_LLM_ARMS)
    assert arms["never"]["balanced_accuracy"] == 0.5
    assert arms["never"]["miss_rate"] == 1.0 and arms["never"]["interruption_rate"] == 0.0
    assert arms["always"]["balanced_accuracy"] == 0.5
    assert arms["always"]["miss_rate"] == 0.0 and arms["always"]["interruption_rate"] == 1.0
    assert arms["gate"]["balanced_accuracy"] < 1.0, "满分=金标照算法写的自证，回退了"


def test_clock_trio_offline():
    """时钟三态（公式锁）：同一档案只挪决策钟——次日沉默 / 四周后开口 / 四周后
    但一小时前刚提醒则冷却。days_since_last 固定 1，年龄由 now 实算成 28 天。"""
    by_id = {c["id"]: c for c in STITCH_CASES}
    assert _gate_decide(by_id["clock_day1_silent"]).reason == "below_tau"
    day28 = _gate_decide(by_id["clock_day28_stitch"])
    assert day28.should_stitch is True and day28.evidence_topic == "欧姆定律"
    cd = _gate_decide(by_id["clock_day28_cooldown"])
    assert cd.should_stitch is False and cd.reason == "cooldown"


def test_gate_evidence_selection_on_multi_gap():
    """次级指标抽验：多缺口竞争的证据选择（快忘的优先 / 更薄弱的优先）。"""
    by_id = {c["id"]: c for c in STITCH_CASES}
    assert _gate_decide(by_id["multi_gap_stale_wins"]).evidence_topic == "三端变阻器的应用"
    assert _gate_decide(by_id["multi_gap_weaker_wins"]).evidence_topic == "线性电路"
    assert _gate_decide(by_id["pos_recurrence_beats_gap"]).evidence_topic == "叠加定理"


def test_forgetting_coupling_anchor_cases():
    """遗忘耦合锚点：5 天沉默 / 7 天开口 / 昨天刚练沉默（金标=教师已点头方向）。"""
    by_id = {c["id"]: c for c in STITCH_CASES}
    assert _gate_decide(by_id["fc_gap_five_days"]).reason == "below_tau"
    assert _gate_decide(by_id["fc_gap_seen_yesterday"]).reason == "below_tau"
    assert _gate_decide(by_id["fc_gap_week_stale"]).should_stitch is True
