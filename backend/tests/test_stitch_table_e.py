"""表E评测集（stitch_cases_table_e）：完整性 + 防自证断言。

防自证是本测试的核心：金标=教师规则，门控是数值实现，两者**必须**存在偏离
（冷却类漏提醒）——满分才是危险信号（退回「金标照算法写」的老路）。断言
gate 均衡准确率 < 1.0 且冷却例恰为 FN，防止日后有人把争议用例改没。
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
    """27 例；每例 teacher_rule 是非空句子（教师可点头/摇头）；金标与证据自洽。"""
    assert len(STITCH_CASES) == 27
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
    """闭包抽验（BFS 核心与生产 get_prereq_predecessors 共用 prereq_closure）：锚点先修。"""
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
    """防自证主断言：gate 均衡准确率 < 1.0；never/always 两基线恒 0.5（口径锚）。

    runner 的 run() 是 async，这里用 asyncio 直接驱动（与 test_eval_memory_runners
    同款做法），零 LLM 三臂不碰网络。
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
    assert arms["gate"]["interruption_rate"] == 0.0, "干净负例上门控不应打扰"


def test_gate_agrees_on_remaining_cases():
    """除冷却两例外，门控与其余 25 例金标一致（偏离只发生在设计争议处）。"""
    mismatches = {
        c["id"]: (c["gold"]["stitch"], pred.should_stitch)
        for c in STITCH_CASES
        if (pred := _gate_decide(c)).should_stitch != c["gold"]["stitch"]
    }
    assert mismatches == {"cooldown_1h": (True, False), "cooldown_20h": (True, False)}, (
        f"偏离面变了（新增/丢失偏离例）：{mismatches}——若金标或门控口径有意变更，"
        "请同步更新本断言与论文失败分析"
    )


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
