"""「知识来源」卡断链修复测试：rag 命中 → tool_result SSE 事件回带 chunks。

根因（三段断链，修复分三段、全在后端，前端零改动——生产 bundle 已带
ChatWindow event.chunks → SourceCard 消费路径，api.ts 另有 contexts 兼容路）：
1. llamaindex_pg.retrieve_context 只回带 last_sources（文件名列表），分数/正文丢弃；
2. _execute_rag 组装的 sources 不含逐条命中内容（test_rag_tool_signal 只锁了 file_paths）；
3. tool_dispatch 发 SSE 只带 {type, tool, content}——sources 整个被丢，前端
   event.chunks 恒空，知识来源卡在生产从不渲染（v6 演示旁白宣称「标注出处」
   但画面从未出现卡片即此断链，代码注释「前端来源卡可溯源」的意图从未生效）。

chunks 形状对齐前端 RagChunk：{content≤300, source, score}。score 用名次启发
（1-idx*0.01，与前端 api.ts 旧兜底同式）——RRF 融合分数量级 ~0.03，直接展示
会变成「3% 匹配」，名次徽章才是卡片要传达的信息。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.agent.tool_registry import _execute_rag
from core.agent.tool_protocol import ToolResult
from core.agentic.tool_dispatch import dispatch_tool_calls
from core.agentic.types import ToolCall


# ── 3. dispatch 层：chunks 必须上 SSE 事件 ────────────────────────────────────


def _tc(name: str) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments={"query": "q"}, arguments_str="{}")


def _stream() -> MagicMock:
    s = MagicMock()
    s.emit = AsyncMock()
    return s


def _tool_result_events(stream: MagicMock) -> list[dict]:
    return [c.args[0] for c in stream.emit.await_args_list
            if isinstance(c.args[0], dict) and c.args[0].get("type") == "tool_result"]


async def test_dispatch_emits_chunks_on_tool_result(monkeypatch):
    """rag 命中（sources 带 chunks）→ tool_result 事件原样回带 chunks（来源卡数据源）。"""
    chunks = [{"content": "另一个电源需从电路中取出，空出的两点用导线连起来",
               "source": "实验四 叠加定理", "score": 1.0}]

    async def _fake_execute(tool_name, **kw):
        return ToolResult(content="[证据1] ……", success=True,
                          sources=[{"type": "rag", "query": "q", "chunks": chunks}])

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)
    stream = _stream()
    await dispatch_tool_calls([_tc("rag")], course_id="c1", enabled_tools=["rag"], stream=stream)

    events = _tool_result_events(stream)
    assert events, "未发出 tool_result 事件"
    assert events[0].get("chunks") == chunks


async def test_dispatch_no_chunks_key_without_sources(monkeypatch):
    """无 sources 的工具（web_search 等）→ 事件不含 chunks 键（不是空列表，避免假卡）。"""

    async def _fake_execute(tool_name, **kw):
        return ToolResult(content="搜索命中：xxx", success=True)

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)
    stream = _stream()
    await dispatch_tool_calls([_tc("web_search")], course_id="c1",
                              enabled_tools=["web_search"], stream=stream)

    events = _tool_result_events(stream)
    assert events and "chunks" not in events[0]


# ── 2. 工具层：_execute_rag 把 retriever.last_chunks 装进 sources ─────────────


class _FakePgRetriever:
    """模拟 pg 检索器：带 last_chunks（pg 路实现）；lightrag 无此属性（getattr 容错）。"""

    def __init__(self, last_chunks: list[dict] | None) -> None:
        self.last_sources = ["电路分析基础实验教案.docx"]
        if last_chunks is not None:
            self.last_chunks = last_chunks
        self.calls: list[str] = []

    async def retrieve_context(self, **kwargs):
        self.calls.append("retrieve")
        return "[证据1｜来源: 电路分析基础实验教案.docx] 叠加定理实验步骤……"


def _patch_registry(monkeypatch, ready: set[str], retriever) -> None:
    async def _ready(_cid):
        return set(ready)

    monkeypatch.setattr("core.agent.tool_registry._get_ready_backends", _ready)
    monkeypatch.setattr("core.rag.get_retriever", lambda name: retriever)


async def test_execute_rag_sources_carry_chunks(monkeypatch):
    """pg 路：retriever.last_chunks → sources[0].chunks 透传（命中正文+来源+名次分）。"""
    last_chunks = [
        {"content": "另一个电源需从电路中取出", "source": "实验四 叠加定理", "score": 1.0},
        {"content": "表针反偏应反接测量取负值", "source": "实验四 叠加定理", "score": 0.99},
    ]
    pg = _FakePgRetriever(last_chunks)
    _patch_registry(monkeypatch, ready={"llamaindex_pg"}, retriever=pg)

    result = await _execute_rag(course_id="c1", query="叠加定理 单独作用")

    assert result.success is True
    assert result.sources[0]["chunks"] == last_chunks
    assert result.sources[0]["file_paths"] == ["电路分析基础实验教案.docx"]


async def test_execute_rag_without_last_chunks_unchanged(monkeypatch):
    """lightrag 路（无 last_chunks 属性）→ sources 不带 chunks 键，行为与旧版一致。"""

    class _NoAttrRetriever(_FakePgRetriever):
        def __init__(self):  # last_chunks=None → 不设该属性（lightrag 同形）
            super().__init__(None)

    _patch_registry(monkeypatch, ready={"lightrag"}, retriever=_NoAttrRetriever())
    result = await _execute_rag(course_id="c1", query="q")

    assert result.success is True
    assert "chunks" not in result.sources[0]


# ── 4. 主路径：kb_seed 预检索直答同样回带 chunks ──────────────────────────────


async def test_kb_seed_hit_emits_chunks():
    """kb_seed 预检索命中（默认主路径）→ tool_result 事件同样回带 chunks。

    kb_seed 材料够时第 1 轮直接作答、loop 不再调 rag——tool_dispatch 的转发在此
    场景永不触发，来源卡能否渲染取决于本路径（复审 P1）。
    """
    from core.capabilities.chat_pipeline import retrieve_kb_seed
    from core.context import UnifiedContext
    from core.stream_bus import StreamBus

    chunks = [{"content": "另一个电源需从电路中取出", "source": "实验四 叠加定理", "score": 1.0}]
    ctx = UnifiedContext(course_id="c1", user_id="u1",
                         user_message="叠加定理 单独作用", enabled_tools=["rag"])
    bus = StreamBus()
    result = ToolResult(content="[证据1] ……", success=True,
                        sources=[{"type": "rag", "query": "q", "chunks": chunks}])
    with patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=result)):
        out = await retrieve_kb_seed(ctx, bus, header="[知识库预检索]")

    assert out  # 命中注入
    if not bus._closed:  # 先关闭再 drain（对齐 test_kb_seed._events：开着永远等新事件）
        await bus.close()
    events = [e.to_dict() async for e in bus.subscribe()]
    tr = [e for e in events if e.get("type") == "tool_result"]
    assert tr and tr[0].get("chunks") == chunks


async def test_chunks_survive_stream_event_roundtrip():
    """emit(dict) → StreamEvent.from_dict → to_dict 全链保真：chunks 字段不丢。

    stream.py 自述三次「新增事件 type/字段在本层静默降级/丢失」事故——本断言把
    chunks 锁进序列化层回归，防 from_dict 改白名单时测试全绿而生产断链。
    """
    from core.stream import StreamEvent

    chunks = [{"content": "证据摘录", "source": "实验四 叠加定理", "score": 1.0}]
    payload = {"type": "tool_result", "tool": "rag", "content": "x", "chunks": chunks}

    roundtripped = StreamEvent.from_dict(payload).to_dict()

    assert roundtripped.get("type") == "tool_result"
    assert roundtripped.get("chunks") == chunks


# ── 1. 检索器层：retrieve_context 回带 last_chunks ────────────────────────────


class TestRetrieveContextLastChunks:
    async def test_last_chunks_stashed_from_fuse(self, monkeypatch):
        """融合命中 → last_chunks：正文剥【来源】前缀截 300 字、来源取章节名、名次分、最多 4 条。"""
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = LlamaIndexRetriever()
        fused = [
            {"chunk_id": f"n{i}", "content": body, "score": 0.03 * (5 - i), "file_path": ""}
            for i, body in enumerate([
                "【章节: 实验四 叠加定理】\n一个电源单独作用时，另一个电源需从电路中取出，并将空出的两点用导线连起来。",
                "【章节: 实验四 叠加定理】\n表针反偏表示实际方向与参考方向相反，应反接测量，数值取负。",
                "【章节: 实验三 电阻串并联电路】\n要求误差在百分之五以内。",
                "【来源: 教案.docx】\n退路：文件名来源。",
                "第五条不应进入 last_chunks（截 4）。",
            ])
        ]

        async def fake_fuse(_cid, _q, _k):
            return fused

        monkeypatch.setattr(retriever, "_fuse", fake_fuse)
        await retriever.retrieve_context(course_id="c1", query="叠加定理", top_k=5)

        lc = retriever.last_chunks
        assert len(lc) == 4                                  # 截 4 条
        assert lc[0]["source"] == "实验四 叠加定理"           # 来源优先取章节名
        assert lc[0]["content"].startswith("一个电源单独作用")  # 剥掉【章节】前缀
        assert all(len(c["content"]) <= 300 for c in lc)
        assert lc[0]["score"] == 1.0 and lc[1]["score"] == 0.99  # 名次分，非 RRF 原始分
        assert lc[3]["source"] == "教案.docx"                 # 无章节退化 basename

    async def test_last_chunks_cleared_on_empty(self, monkeypatch):
        """空命中 → last_chunks 清空（不残留上一轮，与 last_sources 同语义）。"""
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = LlamaIndexRetriever()
        retriever.last_chunks = [{"content": "残留", "source": "旧", "score": 1.0}]

        async def fake_fuse(_cid, _q, _k):
            return []

        monkeypatch.setattr(retriever, "_fuse", fake_fuse)
        ctx = await retriever.retrieve_context(course_id="c1", query="q", top_k=5)

        assert ctx == ""
        assert retriever.last_chunks == []
