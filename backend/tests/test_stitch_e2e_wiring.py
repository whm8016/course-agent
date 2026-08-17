"""表E e2e 接线自检（CI 离线）：打桩 embed 单例跑完整 run()。

只伪造 embedding API 面——query → 该例 matched_topic 的快照向量（与自身余弦
恰为 1 → 确定性命中；neg_unknown 与凭据探针 → 零向量 → cosine 0 → unknown）。
匹配逻辑、库内时间、复发计数、决策全部走生产真路径（stitch_for_turn 全链）。
断言只锁机制（命中、冷却 reason、基线锚），**不钉 gap 臂/三态的 e2e 结果**：
快照真难度从 b 起折挪 theta，C1 入闸随 b 走——那是测量，不是 bug。
"""
from __future__ import annotations

import pytest


class _FakeEmbed:
    """query → v3 快照里对应 matched_topic 的 embedding；未登记的 query → 零向量。"""

    def __init__(self, query_to_emb: dict[str, list[float] | None], dim: int):
        self._map = query_to_emb
        self._dim = dim

    async def _aget_query_embedding(self, query: str) -> list[float]:
        emb = self._map.get(query)
        return list(emb) if emb else [0.0] * self._dim


@pytest.fixture
async def patched_embed(monkeypatch):
    """先打桩、再让测试调 run()——顺序要紧：run() 开头的凭据探针否则先去打真网。
    桩点 core.rag.llamaindex.pg_store._embed_model：get_embed_model() 见非 None
    单例直接返回（pg_store.py:146-148），stitch_for_turn 与探针同走此 seam，
    生产代码零侵入。"""
    from scripts.eval_memory import run_stitch_e2e
    from scripts.eval_memory.stitch_cases_table_e import STITCH_CASES

    snap = run_stitch_e2e.load_graph_embedded()
    emb_by_topic = {t["topic_id"]: t["embedding"] for t in snap["topics"]}
    dim = len(next(iter(emb_by_topic.values())))
    # 同一 query 在多例中出现时 matched_topic 相同（case 集性质），dict 折叠无损
    query_to_emb = {c["query"]: emb_by_topic.get(c["matched_topic"]) for c in STITCH_CASES}
    fake = _FakeEmbed(query_to_emb, dim)
    monkeypatch.setattr("core.rag.llamaindex.pg_store._embed_model", fake)
    yield


async def test_stitch_e2e_full_wiring(patched_embed):
    from scripts.eval_memory import run_stitch_e2e

    summary = await run_stitch_e2e.run()

    assert summary["suite"] == "table_e_e2e"
    assert summary["cases"] == 30
    arms = summary["arms"]
    assert set(arms) == {"never", "always", "gate"}
    assert arms["never"]["balanced_accuracy"] == 0.5
    assert arms["always"]["balanced_accuracy"] == 0.5
    # 匹配机制：桩向量与自身余弦恰为 1（导出已断言 round 后向量唯一，无平局可翻）
    assert summary["match_misses"] == 0
    # 冷却机制：复发臂（p_need floor 0.7 与 b 无关）下 cooldown_1h/20h 必被冷却压住
    by_id = {r["id"]: r for r in summary["details"]}
    for cid in ("cooldown_1h", "cooldown_20h"):
        assert by_id[cid]["gate"]["reason"] == "cooldown", cid
    # 复发自检已在 run() 内逐例执行（不等会 RuntimeError）——到这里即全过


async def test_stitch_e2e_seeding_deterministic(patched_embed):
    """同一进程重跑幂等：_seed_graph/_seed_case 先清后种，第二轮结果不变。"""
    from scripts.eval_memory import run_stitch_e2e

    first = await run_stitch_e2e.run()
    second = await run_stitch_e2e.run()
    key = lambda r: (r["id"], r["gate"]["stitch"], r["gate"]["reason"], r["match_ok"])
    assert [key(r) for r in first["details"]] == [key(r) for r in second["details"]]
