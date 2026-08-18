"""course_topic_builder 抽取链单测（纯函数 + mock LLM，零真实调用）。

课文锚定边主循环（对齐 LightRAG extract_entities / KGGen get_relations）：
- _parse_reply：块回复原始解析（topics/edges 字段清洗）
- _validate_block_edges：闭集端点（全局已抽主题，旧点合法当端点）+ 引句子串 + strength 钳位
- extract_block_graph：单块多轮（首轮+gleaning）汇总、统一校验、两试皆败返 None
- extract_graph_from_chunks：块序闭集累积（跨章边来源）、failed_sections、dropped 计数
- merge_synonymous：去重 + 别名表（归一 label → topic_id，边端点重映射依据）
- resolve_synonyms_llm：LLM 节点归并（embedding 中间带 + 包含对 blocking、闭集校验、
  canonical 实验章节优先、失败降级仅向量去重）；归并后 alias 喂 finalize_edges 自环消失
- load_course_chunks：图注孤儿块两代格式读入即丢、正文剥 [图: …]、空 section 退化用
  chunk_sources 文件名分组（不再落「未知章节」）
- build_topic_graph 接线：merge → resolve → assemble，stats 带 topics_merged_* 两键
- finalize_edges：别名重映射 / 自环剔除 / (src,dst) 去重取高置信（有证据优先）
- assemble_edges：删环 → 闭包 → stats 五键（跨章口径只数直接边）
- break_cycles / close_transitive：DAG 保证与传递闭包
- 次级：_llm_json 契约（非 JSON 重试一次 / 传输失败吞 LLMRetryError 返 None）
- 输出预算三防：known 清单渲染封顶（闭集不受影响）/ finish=length 不原样重试 /
  gleaning 输入守卫（首轮超限跳过补抽）+ 数量上限随 prompt 下发
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.memory.course_topic_builder import (
    _GLEANING_FIRST_ROUND_CHAR_CAP,
    _KNOWN_LIST_RENDER_CAP,
    _MAX_EDGES_PER_BLOCK,
    _MAX_TOPICS_PER_BLOCK,
    _candidate_pairs,
    _llm_json,
    _parse_reply,
    _render_known,
    _validate_block_edges,
    assemble_edges,
    break_cycles,
    clear_rebuild_pending,
    close_transitive,
    extract_block_graph,
    extract_graph_from_chunks,
    finalize_edges,
    load_course_chunks,
    mark_rebuild_pending,
    merge_synonymous,
    read_failed_sections,
    resolve_synonyms_llm,
)


def _resp(content: str, *, reasoning_content=None, finish_reason="stop"):
    """构造 _llm_json 能吃的假 LLM 响应（无 usage 属性 → track 静默跳过）。"""
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning_content),
        finish_reason=finish_reason,
    )])


def _fake_llm(responses: list):
    """假 LLM：str 项经 _resp 包一层（默认 finish=stop）；传 _resp 响应对象则原样用
    （finish_reason 混排场景）。"""
    llm = SimpleNamespace()
    llm.chat = SimpleNamespace()
    llm.chat.completions = SimpleNamespace()
    llm.chat.completions.create = AsyncMock(side_effect=[
        _resp(r) if isinstance(r, str) else r for r in responses
    ])
    return llm


def _topic(label: str, emb=None, sec="第1章", order=0):
    return {
        "topic_id": label, "label": label, "definition": f"{label}的定义",
        "source_section": sec, "order_idx": order, "embedding": emb,
    }


# ---------------------------------------------------------------------------
# _parse_reply：块回复原始解析
# ---------------------------------------------------------------------------

def test_parse_reply_filters_and_cleans():
    data = {
        "topics": [
            {"label": " 戴维南等效 ", "definition": " 等效电源定理 "},
            {"label": "", "definition": "空 label 丢"},
            "junk",
        ],
        "edges": [
            {"src": "元件伏安特性", "dst": "等效内阻测量", "strength": 0.9, "evidence": " 原句 "},
            {"src": "", "dst": "等效内阻测量", "strength": 0.5, "evidence": "x"},  # 缺 src 丢
            {"src": "只有src"},                                                   # 缺 dst 丢
        ],
    }
    topics, edges = _parse_reply(data)
    assert topics == [{"label": "戴维南等效", "definition": "等效电源定理"}]
    assert edges == [{
        "src": "元件伏安特性", "dst": "等效内阻测量",
        "strength": 0.9, "evidence": "原句",
    }]


def test_parse_reply_non_dict_returns_empty():
    assert _parse_reply(None) == ([], [])
    assert _parse_reply(["列表不合法"]) == ([], [])


# ---------------------------------------------------------------------------
# _validate_block_edges：闭集端点 + 引句子串 + strength 钳位
# ---------------------------------------------------------------------------

def test_validate_block_edges_closed_set_and_evidence():
    # 闭集 = 已抽旧点 ∪ 本块新点：旧点（跨章）合法当端点，未知端点丢边计数
    closed = {"元件伏安特性", "等效内阻测量"}
    text = "在元件伏安特性的基础上测量等效内阻。"
    raw = [
        {"src": "元件伏安特性", "dst": "等效内阻测量", "strength": 0.9,
         "evidence": "在元件伏安特性的基础上测量等效内阻"},          # 全合法
        {"src": "不存在点", "dst": "等效内阻测量", "strength": 0.8, "evidence": "x"},  # 闭集外 → 丢
        {"src": "等效内阻测量", "dst": "元件伏安特性", "strength": "崩",
         "evidence": "编造的引句"},                                   # 端点合法；strength 兜底；引句非子串 → 置空
        {"src": "等效内阻测量", "dst": "元件伏安特性", "strength": 1.7, "evidence": ""},  # 钳位到 1.0
        {"src": "元件伏安特性", "dst": "等效内阻测量", "strength": float("nan"), "evidence": ""},  # NaN 钻不过钳位 → 兜底
    ]
    out, dropped = _validate_block_edges(raw, text, closed)
    assert dropped == 1
    assert len(out) == 4
    assert out[0] == {"src_label": "元件伏安特性", "dst_label": "等效内阻测量",
                      "confidence": 0.9, "evidence": "在元件伏安特性的基础上测量等效内阻"}
    assert out[1]["confidence"] == 0.75 and out[1]["evidence"] == ""  # 软伤：去引句留边
    assert out[2]["confidence"] == 1.0
    assert out[3]["confidence"] == 0.75  # NaN 若入库会把教师面板 JSON 打成非法字面量


# ---------------------------------------------------------------------------
# extract_block_graph：单块抽取（首轮 + gleaning 统一校验）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_block_graph_merges_gleaning():
    known = {"旧点": "旧点"}
    text = "基于旧点讲新点。补抽点推出新点。"
    reply1 = {"topics": [{"label": "新点", "definition": "d"}],
              "edges": [{"src": "旧点", "dst": "新点", "strength": 0.8, "evidence": "基于旧点讲新点"}]}
    reply2 = {"topics": [{"label": "补抽点", "definition": "d2"}, {"label": "新点", "definition": "重复"}],
              "edges": [{"src": "补抽点", "dst": "新点", "strength": 0.7, "evidence": "补抽点推出新点"}]}
    llm = _fake_llm([json.dumps(reply1, ensure_ascii=False), json.dumps(reply2, ensure_ascii=False)])
    topics, edges, dropped = await extract_block_graph(
        "c1", "第2章", text, known, llm, "m", gleaning=1,
    )
    # gleaning 补抽并入；重复「新点」与旧点「旧点」不进新主题
    assert [t["label"] for t in topics] == ["新点", "补抽点"]
    assert dropped == 0
    assert [(e["src_label"], e["dst_label"]) for e in edges] == [("旧点", "新点"), ("补抽点", "新点")]


@pytest.mark.asyncio
async def test_extract_block_graph_round1_edge_to_gleaning_topic():
    # 统一校验契约：round1 的边可引用 round2 才补抽出的点（闭集 = known ∪ 全部轮次新主题）
    text = "讲新点。由铺垫点推出新点。"
    reply1 = {"topics": [{"label": "新点", "definition": "d"}],
              "edges": [{"src": "铺垫点", "dst": "新点", "strength": 0.8, "evidence": "由铺垫点推出新点"}]}
    reply2 = {"topics": [{"label": "铺垫点", "definition": "d2"}], "edges": []}
    llm = _fake_llm([json.dumps(reply1, ensure_ascii=False), json.dumps(reply2, ensure_ascii=False)])
    topics, edges, dropped = await extract_block_graph("c1", "第1章", text, {}, llm, "m", gleaning=1)
    assert [t["label"] for t in topics] == ["新点", "铺垫点"]
    assert dropped == 0
    assert [(e["src_label"], e["dst_label"]) for e in edges] == [("铺垫点", "新点")]


@pytest.mark.asyncio
async def test_extract_block_graph_gleaning_failure_keeps_first_round():
    text = "讲新点。"
    reply1 = {"topics": [{"label": "新点", "definition": "d"}], "edges": []}
    llm = _fake_llm([json.dumps(reply1, ensure_ascii=False), "崩", "还崩"])
    topics, edges, dropped = await extract_block_graph(
        "c1", "第1章", text, {}, llm, "m", gleaning=1,
    )
    assert [t["label"] for t in topics] == ["新点"]
    assert edges == []


@pytest.mark.asyncio
async def test_extract_block_graph_all_fail_returns_none():
    llm = _fake_llm(["崩", "还崩"])
    assert await extract_block_graph("c1", "第1章", "文本", {}, llm, "m", gleaning=0) is None


# ---------------------------------------------------------------------------
# 输出预算三防：清单封顶 / finish=length 快速失败 / gleaning 输入守卫
# ---------------------------------------------------------------------------

def test_render_known_caps_at_recent():
    # 130 条旧点只渲染最近 60 条 + 说明行；封顶只管「给模型看」，校验闭集仍是全量 known
    known = {f"点{i}": f"点{i}" for i in range(130)}
    rendered = _render_known(known, [])
    lines = rendered.splitlines()
    assert "只列最近" in lines[0]  # 首行说明：更早旧点仍可当边端点
    body = [l for l in lines if l.startswith("- ")]
    assert len(body) == _KNOWN_LIST_RENDER_CAP
    assert body[-1] == "- 点129"  # 最近的保留
    assert "- 点0" not in rendered  # 最早的挤出渲染（闭集校验仍认）
    # 不超量：全量渲染、无说明行；空清单：占位说明不变
    assert _render_known({"a": "a"}, []) == "- a"
    assert _render_known({}, []) == "（暂无——本块是第一块）"


@pytest.mark.asyncio
async def test_llm_json_finish_length_fails_fast_no_retry():
    # finish=length = 输出额度用尽被腰斩，同额度原样重试必再截（线上实测 17k 字截两次）：
    # 不烧第二次调用，直接 None（上层记 failed_section）。finish=stop 噪声的重试一次
    # 语义不变，由既有 test_llm_json_two_failures_return_none 锁死——快速失败只针对 length
    llm = _fake_llm([_resp('{"topics": [{"label": "半截', finish_reason="length")])
    assert await _llm_json(llm, "m", "p") is None
    assert llm.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_extract_block_graph_skips_gleaning_on_huge_first_round():
    # LightRAG gleaning 输入守卫：首轮结果超大 → 跳过补抽（省一次肥调用），首轮成果照常返回
    big = {"topics": [{"label": f"知识点{i:03d}", "definition": "定" * 30} for i in range(100)],
           "edges": []}
    assert len(json.dumps(big, ensure_ascii=False)) > _GLEANING_FIRST_ROUND_CHAR_CAP
    llm = _fake_llm([json.dumps(big, ensure_ascii=False)])
    topics, edges, dropped = await extract_block_graph(
        "c1", "第1章", "文本", {}, llm, "m", gleaning=1,
    )
    assert llm.chat.completions.create.await_count == 1  # 只有首轮，没有补抽
    assert len(topics) == 100
    assert edges == []


@pytest.mark.asyncio
async def test_extract_prompt_carries_output_caps_and_compact_json():
    # prompt 契约：数量上限 + 单行紧凑指令必须随请求下发（防手改 prompt 时丢防线）
    llm = _fake_llm(['{"topics": [], "edges": []}'])
    await extract_block_graph("c1", "第1章", "文本", {}, llm, "m", gleaning=0)
    prompt = llm.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert f"每块至多 {_MAX_TOPICS_PER_BLOCK} 条" in prompt
    assert f"每块至多 {_MAX_EDGES_PER_BLOCK} 条边" in prompt
    assert "单行紧凑" in prompt


@pytest.mark.asyncio
async def test_extract_prompt_carries_grain_anchors_both_rounds():
    # prompt 契约：粒度锚（3~8 量锚 + 变体归并）首轮与 gleaning 轮都得带——
    # 补抽轮不带量锚会把首轮收粗的粒度又拆细回去（同一防线两轮一致）
    llm = _fake_llm([
        '{"topics": [{"label": "新点", "definition": "d"}], "edges": []}',
        '{"topics": [], "edges": []}',
    ])
    await extract_block_graph("c1", "第1章", "文本", {}, llm, "m", gleaning=1)
    assert llm.chat.completions.create.await_count == 2
    prompts = [
        c.kwargs["messages"][0]["content"]
        for c in llm.chat.completions.create.await_args_list
    ]
    for p in prompts:
        assert "3~8" in p, "量锚（3~8 个）必须随 prompt 下发"
        assert "变体归并" in p, "变体归并指令必须随 prompt 下发"


# ---------------------------------------------------------------------------
# extract_graph_from_chunks：块序闭集累积（跨章边来源）+ failed_sections
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_graph_accumulates_known_across_sections():
    # 第1章抽「旧点」；第2章的边以「旧点」为 src（跨章先修：靠全局闭集合法）；
    # 第3章两试皆败 → failed_sections
    chunks = [
        {"text": "本章教旧点。", "section": "第1章", "order_idx": 0},
        {"text": "基于旧点，本章教新点。", "section": "第2章", "order_idx": 1},
        {"text": "失败章正文。", "section": "第3章", "order_idx": 2},
    ]
    reply1 = {"topics": [{"label": "旧点", "definition": "d"}], "edges": []}
    reply2 = {"topics": [{"label": "新点", "definition": "d2"}],
              "edges": [{"src": "旧点", "dst": "新点", "strength": 0.9, "evidence": "基于旧点，本章教新点"}]}
    llm = _fake_llm([
        json.dumps(reply1, ensure_ascii=False),
        json.dumps(reply2, ensure_ascii=False),
        "崩", "还崩",
    ])
    topics, raw_edges, meta = await extract_graph_from_chunks(
        "c1", chunks, llm, "m", gleaning=0,
    )
    assert [t["label"] for t in topics] == ["旧点", "新点"]
    assert [(e["src_label"], e["dst_label"]) for e in raw_edges] == [("旧点", "新点")]
    assert raw_edges[0]["evidence"] == "基于旧点，本章教新点"
    assert meta["failed_sections"] == ["第3章（第1块）"]
    assert meta["edges_dropped_invalid"] == 0


# ---------------------------------------------------------------------------
# merge_synonymous：去重 + 别名表
# ---------------------------------------------------------------------------

def test_merge_synonymous_keeps_nonnull_embedding():
    topics = [
        _topic("戴维南等效", None, order=0),
        _topic("戴维南等效", [0.1, 0.2], order=1),
    ]
    merged, alias = merge_synonymous(topics)
    assert len(merged) == 1
    assert merged[0]["embedding"] == [0.1, 0.2]
    # 两个重复写法都映射到同一 topic_id（边端点重映射依据）
    assert alias["戴维南等效"] == merged[0]["topic_id"]


def test_merge_synonymous_label_dedup_and_reindex():
    topics = [
        _topic("伏安特性", order=0), _topic("Other", order=1), _topic("伏安特性", order=2),
    ]
    merged, alias = merge_synonymous(topics)
    assert [t["label"] for t in merged] == ["伏安特性", "Other"]
    assert [t["order_idx"] for t in merged] == [0, 1]
    assert alias["伏安特性"] == merged[0]["topic_id"]
    assert alias["other"] == merged[1]["topic_id"]


# ---------------------------------------------------------------------------
# finalize_edges：别名重映射 / 自环剔除 / (src,dst) 去重
# ---------------------------------------------------------------------------

def test_finalize_edges_remap_dedup_selfloop():
    alias = {"a": "a", "b": "b", "a写法2": "a"}  # 「a写法2」并入规范点 a
    raw_edges = [
        {"src_label": "A", "dst_label": "B", "confidence": 0.7, "evidence": ""},
        {"src_label": "A写法2", "dst_label": "B", "confidence": 0.9, "evidence": ""},  # 同 (a,b) 取高置信
        {"src_label": "A", "dst_label": "A写法2", "confidence": 0.8, "evidence": ""},  # 重映射后自环 → 丢
    ]
    edges, dropped = finalize_edges(raw_edges, alias)
    assert dropped == 1
    assert [(e["src"], e["dst"], e["confidence"]) for e in edges] == [("a", "b", 0.9)]


def test_finalize_edges_tie_prefers_evidence():
    alias = {"a": "a", "b": "b"}
    raw_edges = [
        {"src_label": "A", "dst_label": "B", "confidence": 0.8, "evidence": ""},
        {"src_label": "A", "dst_label": "B", "confidence": 0.8, "evidence": "讲义原句"},
    ]
    edges, dropped = finalize_edges(raw_edges, alias)
    assert dropped == 0
    assert len(edges) == 1
    assert edges[0].get("evidence") == "讲义原句"


def test_finalize_edges_direction_conflict_both_kept():
    # 不同块给出相反方向：不在此仲裁，留给 break_cycles 删环删低置信边
    alias = {"a": "a", "b": "b"}
    raw_edges = [
        {"src_label": "A", "dst_label": "B", "confidence": 0.9, "evidence": ""},
        {"src_label": "B", "dst_label": "A", "confidence": 0.3, "evidence": ""},
    ]
    edges, dropped = finalize_edges(raw_edges, alias)
    assert dropped == 0
    assert {(e["src"], e["dst"]) for e in edges} == {("a", "b"), ("b", "a")}


# ---------------------------------------------------------------------------
# assemble_edges：删环 → 闭包 → stats 五键
# ---------------------------------------------------------------------------

def test_assemble_edges_stats_and_cycle():
    # a→b、b→c（跨章）与 c→a 成环 → 删低置信 c→a；闭包补 a→c（prerequisite_inferred）
    topics = [
        _topic("A", sec="第1章", order=0),
        _topic("B", sec="第1章", order=1),
        _topic("C", sec="第5章", order=2),
    ]
    alias = {"a": "A", "b": "B", "c": "C"}
    raw_edges = [
        {"src_label": "A", "dst_label": "B", "confidence": 0.9, "evidence": "原句1"},
        {"src_label": "B", "dst_label": "C", "confidence": 0.8, "evidence": "原句2"},
        {"src_label": "C", "dst_label": "A", "confidence": 0.3, "evidence": ""},
    ]
    edges, stats = assemble_edges(raw_edges, alias, topics, dropped_invalid=2)
    by = {(e["src"], e["dst"]): e for e in edges}
    assert set(by) == {("A", "B"), ("B", "C"), ("A", "C")}
    assert by[("A", "C")]["relation"] == "prerequisite_inferred"
    assert by[("A", "B")]["evidence"] == "原句1"
    assert by[("A", "C")]["evidence"] is None  # 推断边无证据（键恒存在，值为 None）
    # 跨章口径只数直接边：仅 B→C 一条（在闭包后数会被推断边 A→C 虚高到 2）
    assert stats == {
        "text_mined_edges": 3, "edges_dropped_invalid": 2,
        "cycle_broken": 1, "transitive_added": 1, "cross_section_edges": 1,
    }


# ---------------------------------------------------------------------------
# break_cycles / close_transitive
# ---------------------------------------------------------------------------

def _e(src: str, dst: str, conf: float):
    return {"src": src, "dst": dst, "relation": "prerequisite", "confidence": conf}


def test_break_cycles_removes_weakest_edge():
    # A→B→C→A 三元环：删 C→A（最低置信）后成 DAG
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("C", "A", 0.3)]
    out = break_cycles(edges)
    assert {(e["src"], e["dst"]) for e in out} == {("A", "B"), ("B", "C")}
    # 入参不被原地修改
    assert len(edges) == 3


def test_break_cycles_no_cycle_untouched():
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("A", "C", 0.7)]
    assert break_cycles(edges) == edges


def test_close_transitive_fills_and_keeps_direct():
    edges = [_e("A", "B", 0.8), _e("B", "C", 0.6), _e("A", "D", 0.9)]
    out = close_transitive(edges)
    by = {(e["src"], e["dst"]): e["confidence"] for e in out}
    assert by[("A", "C")] == pytest.approx(0.6)   # min(0.8, 0.6) 传递补边
    assert by[("A", "B")] == pytest.approx(0.8)   # 直接边不覆盖
    assert ("D", "B") not in by                    # D 无出边闭包不乱补


def test_close_transitive_direct_edge_not_upgraded():
    # A→C 已有直接边 0.5，经 B 的路径 min(0.9,0.8)=0.8 更高——直接边不被覆盖
    # （课文直接边优先于传递推断）
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("A", "C", 0.5)]
    out = close_transitive(edges)
    by = {(e["src"], e["dst"]): e["confidence"] for e in out}
    assert by[("A", "C")] == pytest.approx(0.5)
    assert len(out) == 3   # 无新边补入


# ---------------------------------------------------------------------------
# _llm_json 契约
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_json_retries_once_then_succeeds():
    llm = _fake_llm(["不是json", '{"ok": 1}'])
    data = await _llm_json(llm, "m", "prompt")
    assert data == {"ok": 1}
    assert llm.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_llm_json_two_failures_return_none():
    llm = _fake_llm(["崩", "还崩"])
    assert await _llm_json(llm, "m", "prompt") is None
    assert llm.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_llm_json_transport_error_returns_none():
    # 传输层异常经 with_retry_and_circuit 包装成 LLMRetryError：不可重试错误一波即弃、
    # 不再发第二波（熔断打开时重试必失败），返回 None（不向上抛——上层记 failed_section）
    from core.llm.reliability import get_llm_circuit_breaker

    breaker = get_llm_circuit_breaker("topic_build")
    breaker.reset()  # 模块级注册表跨测试共享，进出都复位防污染
    llm = _fake_llm([])
    llm.chat.completions.create = AsyncMock(side_effect=ValueError("bad api key"))
    assert await _llm_json(llm, "m", "prompt") is None
    assert llm.chat.completions.create.await_count == 1
    breaker.reset()


@pytest.mark.asyncio
async def test_llm_json_disables_thinking_for_flash():
    """抽取关思考：flash 注入 thinking=disabled，只解析 content。"""
    llm = _fake_llm(['{"ok": 1}'])
    llm.base_url = "https://api.deepseek.com"
    data = await _llm_json(llm, "deepseek-v4-flash", "prompt")
    assert data == {"ok": 1}
    kwargs = llm.chat.completions.create.await_args.kwargs
    assert kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_llm_json_ignores_reasoning_content():
    """只要最终 content；思考字段里即使有 JSON 也不用。"""
    llm = _fake_llm([])
    llm.chat.completions.create = AsyncMock(
        return_value=_resp("", reasoning_content='{"ok": 1}', finish_reason="length"),
    )
    assert await _llm_json(llm, "m", "prompt") is None


def test_clear_failed_sections_drops_stale_banner(tmp_path, monkeypatch):
    """重建入队后 GET 不得继续读上一轮 failed_sections（否则教师面板立刻误报抽取失败）。"""
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    path = tmp_path / "course_c1_topics.json"
    path.write_text(json.dumps({
        "course_id": "c1",
        "topics": [],
        "edges": [],
        "failed_sections": ["未知章节（第1块）"],
    }, ensure_ascii=False), encoding="utf-8")
    assert read_failed_sections("c1") == ["未知章节（第1块）"]
    mark_rebuild_pending("c1")
    assert read_failed_sections("c1") == []
    assert json.loads(path.read_text(encoding="utf-8"))["failed_sections"] == ["未知章节（第1块）"]
    m.export_json("c1", [], [], failed_sections=["真失败（第2块）"])
    assert not (tmp_path / "course_c1_topics.pending").exists()
    assert read_failed_sections("c1") == ["真失败（第2块）"]


def test_clear_failed_sections_missing_file_is_noop(tmp_path, monkeypatch):
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    mark_rebuild_pending("no_such")
    assert read_failed_sections("no_such") == []
    clear_rebuild_pending("no_such")
    assert read_failed_sections("no_such") == []


# ---------------------------------------------------------------------------
# 构建进度旁路文件（pending 空标记升级 JSON）：queued → extracting → finalizing →
# 导出删除；失败写 error；worker 硬死没人写 error 时靠 ts 派生 stalled
# ---------------------------------------------------------------------------

@pytest.fixture
def ctb(tmp_path, monkeypatch):
    """course_topic_builder 模块，JSON_DIR 旁路到 tmp_path（进度/artifact 全落测试目录）。"""
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    return m


def test_build_status_lifecycle(ctb):
    """入队 → 抽取 → 导出全链：GET 的 build 字段逐态可读，导出后归 None。"""
    assert ctb.read_build_status("c1") is None  # 无文件 = 无进行中构建
    ctb.mark_rebuild_pending("c1")
    assert ctb.read_build_status("c1")["status"] == "queued"
    ctb.write_build_progress("c1", status="extracting", done=2, total=5, section="第3章")
    st = ctb.read_build_status("c1")
    assert (st["status"], st["done"], st["total"], st["section"]) == ("extracting", 2, 5, "第3章")
    ctb.export_json("c1", [], [])  # 成功导出删 pending → 构建结束
    assert ctb.read_build_status("c1") is None


def test_build_status_stalled_and_error(ctb, tmp_path):
    """非终态超 15 分钟无更新一律改报 stalled（含 finalizing：收尾阶段 worker 死亡
    同样没人写 error）；error 是终态不改写。"""
    import time as _time

    pending = tmp_path / "course_c1_topics.pending"
    pending.write_text(json.dumps({
        "status": "extracting", "done": 1, "total": 9, "ts": _time.time() - 901,
    }), encoding="utf-8")
    st = ctb.read_build_status("c1")
    assert st["status"] == "stalled" and st["done"] == 1 and st["total"] == 9
    pending.write_text(json.dumps({
        "status": "queued", "ts": _time.time() - 901,
    }), encoding="utf-8")
    assert ctb.read_build_status("c1")["status"] == "stalled"  # 排队卡死同样提示
    pending.write_text(json.dumps({
        "status": "finalizing", "done": 9, "total": 9, "ts": _time.time() - 901,
    }), encoding="utf-8")
    assert ctb.read_build_status("c1")["status"] == "stalled"  # 收尾（embedding/入库）卡死同样兜底
    pending.write_text(json.dumps({
        "status": "error", "error": "boom", "ts": _time.time() - 901,
    }), encoding="utf-8")
    st = ctb.read_build_status("c1")
    assert st["status"] == "error" and st["error"] == "boom"
    pending.write_text("{写到一半被读", encoding="utf-8")
    assert ctb.read_build_status("c1") is None  # 损坏容错：纯展示字段不打断 GET


@pytest.mark.asyncio
async def test_extract_writes_block_progress(ctb, monkeypatch):
    """抽取循环逐块写进度：total 抽前预计算，done 含失败块（已处理≠已成功），收尾 finalizing。"""
    writes: list[dict] = []
    real_write = ctb.write_build_progress
    monkeypatch.setattr(ctb, "write_build_progress", lambda cid, **f: (
        writes.append(f), real_write(cid, **f),
    )[1])
    chunks = [
        {"text": "本章教旧点。", "section": "第1章", "order_idx": 0},
        {"text": "失败块正文。", "section": "第2章", "order_idx": 1},
        {"text": "基于旧点教新点。", "section": "第2章", "order_idx": 2},
    ]
    reply1 = {"topics": [{"label": "旧点", "definition": "d"}], "edges": []}
    reply2 = {"topics": [{"label": "新点", "definition": "d2"}], "edges": []}
    llm = _fake_llm([
        json.dumps(reply1, ensure_ascii=False),
        "崩", "还崩",
        json.dumps(reply2, ensure_ascii=False),
    ])
    topics, _, meta = await ctb.extract_graph_from_chunks(
        "c1", chunks, llm, "m", gleaning=0, max_chars_per_call=3,  # 逐 chunk 独立成块
    )
    assert len(topics) == 2 and meta["failed_sections"] == ["第2章（第1块）"]
    # 每块开抽前写一帧（done=已完成块数），全块处理完写 finalizing（done=total）
    assert [(w["status"], w["done"], w["total"], w.get("section")) for w in writes] == [
        ("extracting", 0, 3, "第1章"),
        ("extracting", 1, 3, "第2章"),
        ("extracting", 2, 3, "第2章"),
        ("finalizing", 3, 3, None),
    ]
    st = ctb.read_build_status("c1")
    assert st["status"] == "finalizing" and st["done"] == 3 and st["total"] == 3


@pytest.mark.asyncio
async def test_build_failure_writes_error_status(ctb, monkeypatch):
    """构建失败（如切块审计缺失）写 error 进度而非删 pending——教师端能看到真实原因，
    且 pending 存在使 read_failed_sections 不回显上一轮的陈旧失败章节。"""

    def _raise(cid):
        raise FileNotFoundError("没有摄入切块审计 JSON")

    monkeypatch.setattr(ctb, "load_course_chunks", _raise)
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: 0)))
    with pytest.raises(FileNotFoundError):
        await ctb.build_topic_graph("c1", db)
    st = ctb.read_build_status("c1")
    assert st["status"] == "error" and "切块审计" in st["error"]
    assert ctb.read_failed_sections("c1") == []  # 陈旧失败不回显


@pytest.mark.asyncio
async def test_topic_graph_payload_carries_build(ctb, monkeypatch):
    """GET topic-graph 的 build 字段直通 read_build_status（teacher/admin 共用一份投影）。"""
    from api.kb_indexing import get_topic_graph_payload
    from core.memory import course_topic_store as store

    monkeypatch.setattr(store, "fetch_course_topics", AsyncMock(return_value=[]))
    monkeypatch.setattr(store, "fetch_course_edges", AsyncMock(return_value=[]))
    assert (await get_topic_graph_payload(None, "c1"))["build"] is None
    ctb.mark_rebuild_pending("c1")
    payload = await get_topic_graph_payload(None, "c1")
    assert payload["build"]["status"] == "queued"
    assert payload["failed_sections"] == []


# ---------------------------------------------------------------------------
# load_course_chunks：图注过滤 + 文件名分组（Phase 1 源头过滤）
# ---------------------------------------------------------------------------

@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    """load_course_chunks 旁路到 tmp 审计目录：真实 settings 实例只改
    paths.ingest_chunks_dir（第一候选命中即不触碰 lightrag 回退路径；不能整个换
    get_settings——ingestion 导入链在 import 期也调它）。"""
    from settings import get_settings

    monkeypatch.setattr(get_settings().paths, "ingest_chunks_dir", str(tmp_path))
    d = tmp_path / "course_c1"
    d.mkdir()
    return d


def _write_audit(d, payload: dict) -> None:
    (d / "latest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_chunks_filters_image_noise(audit_dir):
    """图注孤儿块两代格式读入即丢（旧版块以【图片描述】开头、新版带来源前缀）；
    章内正文剥掉 [图: …]；剥完只剩图注的空块丢。检索侧保留——这里只测抽取读入口。"""
    _write_audit(audit_dir, {"chunks": [
        "【图片描述】\n示波器面板图，左边是触发旋钮",                    # 旧格式孤儿图注 → 丢
        "【来源: 面板图.pdf】\n【图片描述】\n按键功能说明",              # 新格式 → 丢
        "【章节: 实验一 示波器】\n触发方式要点[图: 面板截图描述]正文续",  # 内嵌图注剥掉，正文留
        "[图: 整块只是图注]",                                            # 剥完为空 → 丢
        "[图: 图注被 chunk 边界截断，闭括号在下块",                       # 无闭 ] → 剥到块尾后为空 → 丢
    ]})
    parsed = load_course_chunks("c1")
    assert len(parsed) == 1
    assert parsed[0]["text"] == "触发方式要点正文续"
    assert parsed[0]["section"] == "实验一 示波器"


def test_load_chunks_strips_truncated_img_annotation_tail(audit_dir):
    """图注开 [图: 在本块、闭 ] 在下块：截断尾部剥到块尾，闭块正文保留。"""
    _write_audit(audit_dir, {"chunks": [
        "【章节: 实验一】\n前置要点。[图: 截断的图注没有闭括号",
        "【章节: 实验一】\n后续正文继续",
    ]})
    parsed = load_course_chunks("c1")
    assert [p["text"] for p in parsed] == ["前置要点。", "后续正文继续"]


def test_load_chunks_section_falls_back_to_filename(audit_dir):
    """空 section 退化用 chunk_sources 文件名分组（剥 ::chunk-N 后缀），与
    _build_source_prefix 退化口径一致；老审计无 chunk_sources 才回退「未知章节」。"""
    _write_audit(audit_dir, {
        "chunks": ["裸块甲", "裸块乙"],
        "chunk_sources": ["/kb/大纲.docx::chunk-0", "/kb/教案.docx::chunk-1"],
    })
    parsed = load_course_chunks("c1")
    assert [p["section"] for p in parsed] == ["大纲.docx", "教案.docx"]
    assert [p["text"] for p in parsed] == ["裸块甲", "裸块乙"]
    _write_audit(audit_dir, {"chunks": ["裸块"]})  # version<2 老审计无 sources → 防御回退
    assert load_course_chunks("c1")[0]["section"] == "未知章节"


def test_load_chunks_all_filtered_raises_clear_error(audit_dir):
    """整门课全是图片描述块（如纯扫描件 PDF）：早失败并说真因——不能一路走到
    抽取层报「全部块失败（LLM 不可用？）」误导排查方向。"""
    _write_audit(audit_dir, {"chunks": ["【图片描述】\n图一", "【来源: a.pdf】\n【图片描述】\n图二"]})
    with pytest.raises(ValueError, match="图片描述块"):
        load_course_chunks("c1")


# ---------------------------------------------------------------------------
# resolve_synonyms_llm：节点级 LLM 归并（Phase 2）
# ---------------------------------------------------------------------------

def _rtopic(label: str, sec: str, order: int, emb=None, definition=None):
    """resolve 输入形状 = merge_synonymous 输出（topic_id 已定、order_idx 连续）。"""
    return {
        "topic_id": label, "label": label,
        "definition": definition or f"{label}的定义",
        "source_section": sec, "order_idx": order, "embedding": emb,
    }


@pytest.mark.asyncio
async def test_resolve_merges_band_pair_and_prefers_experiment_source():
    """中间带（0.70~0.88）候选对送 LLM；带外对不进 prompt；canonical 取实验章节来源
    （大纲文件名 rank 最差，order_idx 更小也不保）；alias 两种写法都重定向到规范 id。"""
    # 3D 正交手造向量：cos(A,B)=0.75（带内）、cos(A,C)=cos(B,C)=0（带外）
    topics = [
        _rtopic("示波器的使用", "电路大纲.docx", 0, emb=[1.0, 0.0, 0.0],
                definition="示波器是观察信号波形的通用仪器"),
        _rtopic("双踪示波器的触发方式", "实验一 常用电子仪器", 1,
                emb=[0.75, 0.66144, 0.0], definition="触发"),
        _rtopic("叠加定理", "实验四 叠加定理", 2, emb=[0.0, 0.0, 1.0]),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps(
        {"groups": [["示波器的使用", "双踪示波器的触发方式"]]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 1 and len(out) == 2
    canon = next(t for t in out if t["label"] == "双踪示波器的触发方式")
    assert canon["source_section"] == "实验一 常用电子仪器"  # 实验细点吃掉大纲粗点
    assert canon["definition"] == "示波器是观察信号波形的通用仪器"  # 定义取较长者
    assert alias2["示波器的使用"] == canon["topic_id"]
    assert alias2["双踪示波器的触发方式"] == canon["topic_id"]
    assert alias2["叠加定理"] == "叠加定理"  # 无关点不动
    prompt = llm.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    # 候选对区块只有带内那一对（cos 0.0 的「叠加定理」低于下界不送 LLM，防乱并；
    # Instructions 示例里的「叠加定理」字样不算）
    pairs_block = prompt.split("---候选对---")[1].split("---Output---")[0].strip()
    assert pairs_block == (
        "1. 「示波器的使用」（电路大纲.docx） ↔ 「双踪示波器的触发方式」（实验一 常用电子仪器）"
    )


@pytest.mark.asyncio
async def test_resolve_containment_pair_without_embedding():
    """标签归一互相包含的对无 embedding 也是候选（blocking 双通道）；canonical
    （实验来源）embedding 为 NULL 时吸收被并点的（NULL 进表 _match_topic 直接淘汰）。"""
    topics = [
        _rtopic("叠加定理", "第2章 等效电路", 0, emb=None, definition="线性电路叠加原理"),
        _rtopic("叠加定理的验证", "实验四 叠加定理", 1, emb=[0.1, 0.2], definition="测"),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps(
        {"groups": [["叠加定理", "叠加定理的验证"]]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 1 and len(out) == 1
    canon = out[0]
    assert canon["label"] == "叠加定理的验证"  # 实验章节来源优先
    assert canon["embedding"] == [0.1, 0.2]  # NULL 吸收非空
    assert canon["definition"] == "线性电路叠加原理"  # 取较长


@pytest.mark.asyncio
async def test_resolve_failure_degrades_to_vector_dedup():
    """LLM 两试皆败：原样返回（topics/alias 不动、merged=0），不废整图。"""
    topics = [
        _rtopic("示波器的使用", "电路大纲.docx", 0, emb=[1.0, 0.0, 0.0]),
        _rtopic("双踪示波器的触发方式", "实验一", 1, emb=[0.75, 0.66144, 0.0]),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm(["崩", "还崩"])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 0
    assert out is topics and alias2 is alias
    assert len(out) == 2


@pytest.mark.asyncio
async def test_resolve_ignores_invented_labels_and_empty_groups():
    """闭集校验：LLM 发明的标签丢弃、单已知成员组无事可并、groups 空/垃圾形状不炸。"""
    topics = [
        _rtopic("叠加定理", "第2章", 0, emb=None),
        _rtopic("叠加定理的验证", "实验四", 1, emb=None),
        _rtopic("一阶RC电路的零输入响应", "实验七", 2, emb=None),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps({"groups": [
        ["我发明的标签", "另一个发明"],
        ["只有叠加定理"],
        ["叠加定理", "叠加定理的验证"],
        "垃圾条目",
    ]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 1  # 只有第三个组合法生效
    assert alias2["叠加定理"] == alias2["叠加定理的验证"]


@pytest.mark.asyncio
async def test_resolve_never_merges_non_candidate_pair():
    """代码侧兜底 prompt「宁可不并」：LLM 把从未一起判过的两个标签塞进同一组
    （分属不同候选行）→ 不并——照单全收会把独立考点焊死、掌握度互相污染。"""
    topics = [
        _rtopic("叠加定理", "第2章", 0, emb=None),
        _rtopic("叠加定理的验证", "实验四", 1, emb=None),          # 与上者包含 → 候选
        _rtopic("一阶RC电路的零输入响应", "实验七", 2, emb=None),  # 与谁都非候选
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps({"groups": [
        ["叠加定理", "一阶RC电路的零输入响应"],   # 非候选对 → 拒
    ]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 0
    assert alias2["叠加定理"] != alias2["一阶RC电路的零输入响应"]


@pytest.mark.asyncio
async def test_resolve_transitive_chain_through_candidates():
    """组内传递并：A-B（包含）、B-C（向量带）都是候选 → LLM 一组 [A,B,C] 全并成一点
    （A-C 从未候选也随传递合并——只挡「凭空并」，不挡候选链闭包）。"""
    topics = [
        _rtopic("叠加定理", "第2章", 0, emb=None),
        _rtopic("叠加定理的验证", "实验四", 1, emb=[1.0, 0.0, 0.0]),
        _rtopic("线性叠加原理验证", "实验四", 2, emb=[0.75, 0.66144, 0.0]),  # cos(B)=0.75 带内
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps({"groups": [
        ["叠加定理", "叠加定理的验证", "线性叠加原理验证"],
    ]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 2 and len(out) == 1
    assert len({alias2[t["label"]] for t in topics}) == 1  # 三写法同指一点


@pytest.mark.asyncio
async def test_resolve_hub_claimed_by_two_groups_keeps_fine_topics_apart():
    """P1 回归：同一枢纽标签被 LLM 分别写进两组（[大纲,触发]、[大纲,探头]——各自
    成立）→ 不跨组传递：枢纽只并进与 canonical 余弦最高的那组，两个细点彼此独立。
    全局并查集会把 A-B、A-C 传递成 B-C 同点——「怎么触发」和「探头怎么接」打到同一
    掌握度上。"""
    # cos(大纲,触发)=0.80、cos(大纲,探头)≈0.72 均带内；cos(触发,探头)≈0.62 带外
    topics = [
        _rtopic("示波器的使用", "电路大纲.docx", 0, emb=[1.0, 0.0, 0.0],
                definition="示波器是观察信号波形的通用仪器"),
        _rtopic("双踪示波器的触发方式", "实验一 常用电子仪器", 1, emb=[0.8, 0.6, 0.0]),
        _rtopic("示波器探头补偿", "实验二 仪器校准", 2, emb=[0.72, 0.08, 0.69]),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps({"groups": [
        ["示波器的使用", "双踪示波器的触发方式"],
        ["示波器的使用", "示波器探头补偿"],
    ]}, ensure_ascii=False)])
    out, alias2, merged = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    assert merged == 1 and len(out) == 2  # 3 点 → 2 点：只有大纲粗点被吸收
    assert {t["label"] for t in out} == {"双踪示波器的触发方式", "示波器探头补偿"}
    assert alias2["示波器的使用"] == "双踪示波器的触发方式"  # 并进 cos 0.80 那组
    assert alias2["示波器探头补偿"] == "示波器探头补偿"  # 细点独立不动


def test_candidate_pairs_containment_requires_min_length():
    """P2 回归：归一后短词（「电路」2 字）被包含不算强信号，只走余弦带——否则
    半门课的长标签都和它配对；「叠加定理」（4 字）级包含仍走包含通道。"""
    short = _rtopic("电路", "第1章", 0, emb=[1.0, 0.0, 0.0, 0.0])
    long_ = _rtopic("电路分析基础", "第2章", 1, emb=[0.0, 1.0, 0.0, 0.0])  # cos=0 带外
    assert _candidate_pairs([short, long_], 0.70, 0.88) == []
    a = _rtopic("叠加定理", "第2章", 0, emb=[1.0, 0.0, 0.0, 0.0])
    b = _rtopic("叠加定理的验证", "实验四", 1, emb=[0.0, 1.0, 0.0, 0.0])  # cos=0 带外
    assert _candidate_pairs([a, b], 0.70, 0.88) == [(0, 1)]


@pytest.mark.asyncio
async def test_resolve_alias_feeds_finalize_edges():
    """归并扩 alias 后走既有 finalize_edges：合并对之间的边变自环丢弃、指向外部的
    边端点重映射到规范 id——边去重不必再烧 LLM。"""
    topics = [
        _rtopic("示波器触发", "电路大纲.docx", 0, emb=None),
        _rtopic("双踪示波器触发方式", "实验一", 1, emb=None),
        _rtopic("叠加定理", "实验四", 2, emb=None),
    ]
    alias = {t["label"]: t["topic_id"] for t in topics}
    llm = _fake_llm([json.dumps(
        {"groups": [["示波器触发", "双踪示波器触发方式"]]}, ensure_ascii=False)])
    _, alias2, _ = await resolve_synonyms_llm("c1", topics, alias, llm, "m")
    raw_edges = [
        {"src_label": "示波器触发", "dst_label": "双踪示波器触发方式", "confidence": 0.9, "evidence": ""},
        {"src_label": "示波器触发", "dst_label": "叠加定理", "confidence": 0.8, "evidence": "原句"},
    ]
    edges, dropped = finalize_edges(raw_edges, alias2)
    assert dropped == 1  # 两写法已并成一点 → 自环
    assert [(e["src"], e["dst"]) for e in edges] == [("双踪示波器触发方式", "叠加定理")]


@pytest.mark.asyncio
async def test_extract_prompt_rejects_image_notes_both_rounds():
    """prompt 兜底契约：图注/菜单截图不是知识点，首轮与 gleaning 轮都得带（剥漏兜底）。"""
    llm = _fake_llm(['{"topics": [], "edges": []}', '{"topics": [], "edges": []}'])
    await extract_block_graph("c1", "第1章", "文本", {}, llm, "m", gleaning=1)
    assert llm.chat.completions.create.await_count == 2
    prompts = [c.kwargs["messages"][0]["content"]
               for c in llm.chat.completions.create.await_args_list]
    for p in prompts:
        assert "菜单截图" in p, "图注兜底句必须随 prompt 下发"


# ---------------------------------------------------------------------------
# build_topic_graph 接线：merge → resolve → assemble + stats（Phase 3）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_topic_graph_wires_resolve_and_stats(ctb, monkeypatch):
    """全链 wiring：merge(0) → LLM 归并(1，大纲点并入实验点) → assemble（自环边丢弃）；
    artifact stats 带 topics_merged_embedding / topics_merged_llm 两键。"""
    monkeypatch.setattr(ctb, "load_course_chunks", lambda cid: [
        {"text": "正文", "section": "实验一", "order_idx": 0},
    ])
    monkeypatch.setattr("core.rag.llamaindex.pg_store.get_embed_model", lambda: object())

    async def _fake_extract(cid, chunks, llm, model, embed_model=None, gleaning=1):
        topics = [
            _rtopic("示波器触发", "电路大纲.docx", 0, emb=None),
            _rtopic("双踪示波器触发方式", "实验一", 1, emb=None),
        ]
        raw_edges = [{"src_label": "示波器触发", "dst_label": "双踪示波器触发方式",
                      "confidence": 0.8, "evidence": ""}]  # 合并后变自环 → 丢
        return topics, raw_edges, {"failed_sections": [], "edges_dropped_invalid": 0}

    monkeypatch.setattr(ctb, "extract_graph_from_chunks", _fake_extract)

    async def _fake_llm_json(llm, model, prompt, track=None):
        return {"groups": [["示波器触发", "双踪示波器触发方式"]]}

    monkeypatch.setattr(ctb, "_llm_json", _fake_llm_json)
    db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: 0)),
        add=lambda o: None, commit=AsyncMock(),
    )
    result = await ctb.build_topic_graph("c1", db)
    assert result["status"] == "built" and result["topics"] == 1 and result["edges"] == 0
    artifact = json.loads(
        (ctb.JSON_DIR / "course_c1_topics.json").read_text(encoding="utf-8"))
    assert artifact["stats"]["topics_merged_embedding"] == 0
    assert artifact["stats"]["topics_merged_llm"] == 1
    assert artifact["stats"]["edges_dropped_invalid"] == 1  # 自环边
    assert artifact["topics"][0]["source_section"] == "实验一"  # 章节细点吃掉大纲粗点


@pytest.mark.asyncio
async def test_build_topic_graph_resolve_switch_off(ctb, monkeypatch):
    """settings.llm_resolve_enabled=False：跳过 LLM 归并（不烧调用），仅向量去重。"""
    from settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s.course_topic, "llm_resolve_enabled", False)
    monkeypatch.setattr(ctb, "load_course_chunks", lambda cid: [
        {"text": "正文", "section": "实验一", "order_idx": 0},
    ])
    monkeypatch.setattr("core.rag.llamaindex.pg_store.get_embed_model", lambda: object())

    async def _fake_extract(cid, chunks, llm, model, embed_model=None, gleaning=1):
        topics = [_rtopic("示波器触发", "电路大纲.docx", 0, emb=None),
                  _rtopic("双踪示波器触发方式", "实验一", 1, emb=None)]
        return topics, [], {"failed_sections": [], "edges_dropped_invalid": 0}

    monkeypatch.setattr(ctb, "extract_graph_from_chunks", _fake_extract)
    llm_json_calls = []

    async def _spy_llm_json(llm, model, prompt, track=None):
        llm_json_calls.append(prompt)
        return {"groups": []}

    monkeypatch.setattr(ctb, "_llm_json", _spy_llm_json)
    db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: 0)),
        add=lambda o: None, commit=AsyncMock(),
    )
    result = await ctb.build_topic_graph("c1", db)
    assert result["status"] == "built" and result["topics"] == 2
    assert llm_json_calls == []  # 开关关：一次 LLM 归并调用都不发
    artifact = json.loads(
        (ctb.JSON_DIR / "course_c1_topics.json").read_text(encoding="utf-8"))
    assert artifact["stats"]["topics_merged_llm"] == 0
