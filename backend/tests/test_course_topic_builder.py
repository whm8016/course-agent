"""course_topic_builder 抽取链单测（纯函数 + mock LLM，零真实调用）。

课文锚定边主循环（对齐 LightRAG extract_entities / KGGen get_relations）：
- _parse_reply：块回复原始解析（topics/edges 字段清洗）
- _validate_block_edges：闭集端点（全局已抽主题，旧点合法当端点）+ 引句子串 + strength 钳位
- extract_block_graph：单块多轮（首轮+gleaning）汇总、统一校验、两试皆败返 None
- extract_graph_from_chunks：块序闭集累积（跨章边来源）、failed_sections、dropped 计数
- merge_synonymous：去重 + 别名表（归一 label → topic_id，边端点重映射依据）
- finalize_edges：别名重映射 / 自环剔除 / (src,dst) 去重取高置信（有证据优先）
- assemble_edges：删环 → 闭包 → stats 五键（跨章口径只数直接边）
- break_cycles / close_transitive：DAG 保证与传递闭包
- 次级：_llm_json 契约（非 JSON 重试一次 / 传输失败吞 LLMRetryError 返 None）
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.memory.course_topic_builder import (
    _llm_json,
    _parse_reply,
    _validate_block_edges,
    assemble_edges,
    break_cycles,
    clear_rebuild_pending,
    close_transitive,
    extract_block_graph,
    extract_graph_from_chunks,
    finalize_edges,
    mark_rebuild_pending,
    merge_synonymous,
    read_failed_sections,
)


def _resp(content: str, *, reasoning_content=None, finish_reason="stop"):
    """构造 _llm_json 能吃的假 LLM 响应（无 usage 属性 → track 静默跳过）。"""
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning_content),
        finish_reason=finish_reason,
    )])


def _fake_llm(contents: list[str]):
    llm = SimpleNamespace()
    llm.chat = SimpleNamespace()
    llm.chat.completions = SimpleNamespace()
    llm.chat.completions.create = AsyncMock(side_effect=[_resp(c) for c in contents])
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

def test_build_status_lifecycle(tmp_path, monkeypatch):
    """入队 → 抽取 → 导出全链：GET 的 build 字段逐态可读，导出后归 None。"""
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    assert m.read_build_status("c1") is None  # 无文件 = 无进行中构建
    m.mark_rebuild_pending("c1")
    assert m.read_build_status("c1")["status"] == "queued"
    m.write_build_progress("c1", status="extracting", done=2, total=5, section="第3章")
    st = m.read_build_status("c1")
    assert (st["status"], st["done"], st["total"], st["section"]) == ("extracting", 2, 5, "第3章")
    assert isinstance(st.get("ts"), (int, float))  # stalled 判定依据在读侧透出
    m.export_json("c1", [], [])  # 成功导出删 pending → 构建结束
    assert m.read_build_status("c1") is None


def test_build_status_stalled_and_error(tmp_path, monkeypatch):
    """extracting 超 15 分钟无更新改报 stalled（worker 硬死兜底）；error 是终态不改写。"""
    import time as _time

    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    pending = tmp_path / "course_c1_topics.pending"
    pending.write_text(json.dumps({
        "status": "extracting", "done": 1, "total": 9, "ts": _time.time() - 901,
    }), encoding="utf-8")
    st = m.read_build_status("c1")
    assert st["status"] == "stalled" and st["done"] == 1 and st["total"] == 9
    pending.write_text(json.dumps({
        "status": "queued", "ts": _time.time() - 901,
    }), encoding="utf-8")
    assert m.read_build_status("c1")["status"] == "stalled"  # 排队卡死同样提示
    pending.write_text(json.dumps({
        "status": "error", "error": "boom", "ts": _time.time() - 901,
    }), encoding="utf-8")
    st = m.read_build_status("c1")
    assert st["status"] == "error" and st["error"] == "boom"
    pending.write_text("{写到一半被读", encoding="utf-8")
    assert m.read_build_status("c1") is None  # 损坏容错：纯展示字段不打断 GET


@pytest.mark.asyncio
async def test_extract_writes_block_progress(tmp_path, monkeypatch):
    """抽取循环逐块写进度：total 抽前预计算，done 含失败块（已处理≠已成功），收尾 finalizing。"""
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    writes: list[dict] = []
    real_write = m.write_build_progress
    monkeypatch.setattr(m, "write_build_progress", lambda cid, **f: (
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
    topics, _, meta = await m.extract_graph_from_chunks(
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
    st = m.read_build_status("c1")
    assert st["status"] == "finalizing" and st["done"] == 3 and st["total"] == 3


@pytest.mark.asyncio
async def test_build_failure_writes_error_status(tmp_path, monkeypatch):
    """构建失败（如切块审计缺失）写 error 进度而非删 pending——教师端能看到真实原因，
    且 pending 存在使 read_failed_sections 不回显上一轮的陈旧失败章节。"""
    from core.memory import course_topic_builder as m

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)

    def _raise(cid):
        raise FileNotFoundError("没有摄入切块审计 JSON")

    monkeypatch.setattr(m, "load_course_chunks", _raise)
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: 0)))
    with pytest.raises(FileNotFoundError):
        await m.build_topic_graph("c1", db)
    st = m.read_build_status("c1")
    assert st["status"] == "error" and "切块审计" in st["error"]
    assert m.read_failed_sections("c1") == []  # 陈旧失败不回显


@pytest.mark.asyncio
async def test_topic_graph_payload_carries_build(tmp_path, monkeypatch):
    """GET topic-graph 的 build 字段直通 read_build_status（teacher/admin 共用一份投影）。"""
    from api.kb_indexing import get_topic_graph_payload
    from core.memory import course_topic_builder as m
    from core.memory import course_topic_store as store

    monkeypatch.setattr(m, "JSON_DIR", tmp_path)
    monkeypatch.setattr(store, "fetch_course_topics", AsyncMock(return_value=[]))
    monkeypatch.setattr(store, "fetch_course_edges", AsyncMock(return_value=[]))
    assert (await get_topic_graph_payload(None, "c1"))["build"] is None
    m.mark_rebuild_pending("c1")
    payload = await get_topic_graph_payload(None, "c1")
    assert payload["build"]["status"] == "queued"
    assert payload["failed_sections"] == []
