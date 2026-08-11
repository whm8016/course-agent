"""P1 修复回归：graph_memory 已有节点的 mastery/risk delta 必须能降。

根因：_merge_knowledge_graph 现有节点分支对 mastery_delta/risk_delta 误用 _clamp
（max(0, min(1, x))）-> 负 delta（退步 / risk 改善）被夹成 0 -> mastery 只升不降、
risk 单调上升，仪表盘"高风险知识点"只增不减。

修复：delta 改用 _delta（纯 float 转换不钳制，与 mastery._delta 同语义），最终值仍
_clamp 到 [0,1]。新建节点分支本就用 0.5+delta 正确处理负数，不动。
"""
from core.memory.graph_memory import _merge_knowledge_graph, _node_id

import pytest


def test_negative_mastery_delta_decreases_mastery():
    """退步（mastery_delta<0）必须让 mastery 下降，不被钳成 0。"""
    label = "戴维南定理"
    nid = _node_id("kp", label)
    existing = {
        "nodes": [{"id": nid, "label": label, "mastery": 0.8, "risk": 0.5,
                   "status": "active", "examples": []}],
        "edges": [],
    }
    merged = _merge_knowledge_graph(
        existing, [{"label": label, "entity_id": "ent1", "mastery_delta": -0.3}],
        course_id="c1",
    )
    node = merged["nodes"][0]
    assert node["mastery"] == pytest.approx(0.5), \
        f"mastery 应 0.8-0.3=0.5（钳下限），实际 {node['mastery']}"


def test_negative_risk_delta_decreases_risk():
    """risk 改善（risk_delta<0）必须让 risk 下降，不被钳成 0。"""
    label = "戴维南定理"
    nid = _node_id("kp", label)
    existing = {
        "nodes": [{"id": nid, "label": label, "mastery": 0.5, "risk": 0.7,
                   "status": "active", "examples": []}],
        "edges": [],
    }
    merged = _merge_knowledge_graph(
        existing, [{"label": label, "entity_id": "ent1", "risk_delta": -0.4}],
        course_id="c1",
    )
    node = merged["nodes"][0]
    assert node["risk"] == pytest.approx(0.3), f"risk 应 0.7-0.4=0.3，实际 {node['risk']}"


def test_positive_delta_still_increases():
    """正 delta 行为不变（回归保护）：mastery 进步、risk 加剧仍上升。"""
    label = "节点定理"
    nid = _node_id("kp", label)
    existing = {
        "nodes": [{"id": nid, "label": label, "mastery": 0.4, "risk": 0.2,
                   "status": "active", "examples": []}],
        "edges": [],
    }
    merged = _merge_knowledge_graph(
        existing, [{"label": label, "entity_id": "ent2",
                    "mastery_delta": 0.3, "risk_delta": 0.4}],
        course_id="c1",
    )
    node = merged["nodes"][0]
    assert node["mastery"] == pytest.approx(0.7)
    assert node["risk"] == pytest.approx(0.6)
