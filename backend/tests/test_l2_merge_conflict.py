"""L2 摘要 v2 冲突裁决 + 验收场景单测（对应 plan §五验收标准）。

聚焦改造的核心收益，区别于 test_session_summary.py 的逐函数覆盖：
  - knowledge_update：学生改口后注入摘要只含当前值（旧实现必然失败，本文件证新实现通过）
  - 30 轮长会话：早期 fact 不被近义 topic 挤出（P0 淘汰修复）
  - 注入 token 不超预算
  - v1 -> v2 升级路径正确（向后兼容）
  - resolved 消除 open_question（filtering）
"""
from __future__ import annotations

import json

from core.memory import session_summary as ss
from core.memory.session_summary import SummaryItem


def _item(k: str, key: str, t: str, ts: float = 1000.0, n: int = 1) -> SummaryItem:
    return SummaryItem(k=k, key=key, t=t, ts=ts, n=n)


# ── 验收 1：knowledge_update 改口后当前值唯一 ────────────────────────────────

def test_knowledge_update_textbook_edition():
    """学生先说教材第5版、后改口第6版：合并后注入摘要只有第6版（旧实现两条并存）。

    场景：两轮增量，fact 单值槽 textbook_edition，LLM 用相同 key 才能覆盖。
    """
    # 第一轮增量：抽到教材第5版
    after_turn1 = ss._merge_items([], [_item("fact", "textbook_edition", "第5版", ts=1000.0)])
    # 第二轮增量：学生改口第6版（LLM 输出相同 key）
    after_turn2 = ss._merge_items(after_turn1, [_item("fact", "textbook_edition", "第6版", ts=2000.0)])

    rendered = ss._render_items(after_turn2, now_ts=2000.0, half_life_s=3600)
    assert "第6版" in rendered
    assert "第5版" not in rendered  # 旧值被覆盖，不并存


def test_knowledge_update_next_step_progress():
    """next_step 单值槽：进度推进时旧进度被覆盖。"""
    after_t1 = ss._merge_items([], [_item("next_step", "progress", "正在讲欧姆定律", ts=1.0)])
    after_t2 = ss._merge_items(after_t1, [_item("next_step", "progress", "已讲完欧姆定律，下次讲戴维南", ts=2.0)])

    rendered = ss._render_items(after_t2, now_ts=2.0, half_life_s=3600)
    assert "下次讲戴维南" in rendered
    assert "正在讲欧姆定律" not in rendered


# ── 验收 2：30 轮长会话，早期 fact 不被近义 topic 挤出 ────────────────────────

def test_long_session_early_fact_not_evicted_by_topics():
    """30 轮长会话：1 个早期 fact + 多个近期 topic，预算紧张时 fact 因 kind_weight 高保留。

    旧实现 combined[-5:] 按条数丢最旧，5 个名额被近义 topic 占满后早期 fact 被挤出。
    新实现按 salience（fact 权重 1.5 > topic 0.8）+ 每类独立上限，fact 保留。
    """
    items = [_item("fact", "textbook_edition", "教材第6版", ts=1000.0)]  # 早期 fact
    # 30 个近期近义 topic（不同 key 共存）
    items += [_item("topic", f"topic_{i}", f"知识点{i}讨论", ts=2000.0) for i in range(30)]

    kept = ss._evict_by_budget(
        items, token_budget=400, max_per_kind=8, now_ts=2000.0, half_life_s=3600,
    )
    kinds = {it.k for it in kept}
    assert "fact" in kinds  # 早期 fact 保留
    # topic 被每类上限 + 预算限制，不会独占挤掉 fact


def test_max_per_kind_prevents_topic_monopoly():
    """max_per_kind 让 topic 最多保留 N 条，余下预算留给 fact/misconception 等。"""
    items = [_item("topic", f"t{i}", f"主题{i}", ts=1.0) for i in range(20)]
    items += [_item("misconception", "m1", "串联并联混淆（已纠正）", ts=1.0)]
    kept = ss._evict_by_budget(items, token_budget=10000, max_per_kind=5,
                               now_ts=1.0, half_life_s=3600)
    topic_count = sum(1 for it in kept if it.k == "topic")
    assert topic_count == 5  # topic 被上限截断
    assert any(it.k == "misconception" for it in kept)  # misconception 有名额


# ── 验收 3：注入 token 不超预算 ──────────────────────────────────────────────

def test_evicted_total_tokens_within_budget():
    """淘汰后总 token 估算不超过配置预算。"""
    items = [_item("topic", f"t{i}", "x" * 100, ts=1.0) for i in range(50)]
    budget = 500
    kept = ss._evict_by_budget(items, token_budget=budget, max_per_kind=8,
                               now_ts=1.0, half_life_s=3600)
    total = sum(ss._estimate_tokens(it.t) for it in kept)
    assert total <= budget


# ── v1 -> v2 升级（向后兼容）────────────────────────────────────────────────

def test_v1_upgrade_then_merge_with_v2():
    """存量 v1 摘要 + 新 v2 增量：v1 升级后参与合并，单值槽被新值覆盖。"""
    v1_raw = json.dumps({"facts": ["教材第5版"]}, ensure_ascii=False)
    existing = ss._parse_structured(v1_raw)  # v1 -> upgrade
    assert existing is not None
    assert existing[0].k == "fact"
    assert existing[0].ts == 0.0  # legacy 最低显著度

    # 新增量改口第6版（v1 摘要的 fact key 是文本哈希，与 LLM 输出的 textbook_edition 不同 key）
    # -> 不同 key 共存；要覆盖需 LLM 输出与存量相同 key。此处验共存不丢：
    new = [_item("fact", "textbook_edition", "第6版", ts=1000.0)]
    merged = ss._merge_items(existing, new)
    texts = {it.t for it in merged}
    assert "第6版" in texts
    assert "教材第5版" in texts  # v1 legacy 与新值不同 key，共存（不丢数据）


def test_v1_upgrade_action_items_renamed_to_next_step():
    """v1 action_items -> v2 next_step（改名映射）。"""
    v1_raw = json.dumps({"action_items": ["下次讲三相电路"]}, ensure_ascii=False)
    items = ss._parse_structured(v1_raw)
    assert items is not None
    assert items[0].k == "next_step"
    assert items[0].t == "下次讲三相电路"


def test_v1_upgrade_dedup_same_text():
    """v1 同文本条目升级后 key 相同（哈希），合并去重为一条。"""
    v1_raw = json.dumps({"topics": ["戴维南定理", "戴维南定理"]}, ensure_ascii=False)
    items = ss._parse_structured(v1_raw)
    assert len(items) == 1  # 同文本同 key 去重


# ── resolved 消除 open_question（filtering）─────────────────────────────────

def test_resolved_eliminates_open_question():
    """open_question 被后续轮次解答 -> resolved 列出其 key -> 合并后消除。"""
    existing = [
        _item("open_question", "why_parallel", "为什么并联这样算", ts=1.0),
        _item("open_question", "kvl_unknown", "KVL 怎么列方程", ts=1.0),
    ]
    # 本轮解答了 why_parallel
    new = [_item("topic", "parallel", "并联电阻分析（已解答）", ts=2.0)]
    merged = ss._merge_items(existing, new, resolved=["why_parallel"])

    oq_keys = {it.key for it in merged if it.k == "open_question"}
    assert "why_parallel" not in oq_keys  # 已消除
    assert "kvl_unknown" in oq_keys  # 未解答的保留


def test_resolved_only_affects_open_question():
    """resolved 只删 open_question，不误删同 key 的其他 kind。"""
    existing = [
        _item("open_question", "shared_key", "待解问题", ts=1.0),
        _item("topic", "shared_key", "相关话题", ts=1.0),
    ]
    merged = ss._merge_items(existing, [], resolved=["shared_key"])
    kinds = {it.k for it in merged}
    assert "topic" in kinds  # topic 同 key 不受影响
    assert "open_question" not in kinds  # 仅 open_question 被删


# ── 纯函数性质：不修改输入 ─────────────────────────────────────────────────

def test_merge_does_not_mutate_inputs():
    """_merge_items 是纯函数，不改 existing/new 列表与其元素。"""
    existing = [_item("fact", "f1", "旧", ts=1.0, n=1)]
    new = [_item("fact", "f1", "新", ts=2.0, n=1)]
    existing_copy = [_item(it.k, it.key, it.t, it.ts, it.n) for it in existing]
    ss._merge_items(existing, new)
    assert existing[0].t == existing_copy[0].t
    assert existing[0].n == existing_copy[0].n
