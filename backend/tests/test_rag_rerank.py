"""core/rag/rerank.py 单测：build_rerank_fn 的惰性构造 + rerank_fn 的候选截断/预算/映回语义。

纯 mock——不调真实 DashScope，验证：
- build_rerank_fn 无 key / 开关关闭时返回 None（force 语义）
- _call_rerank_api 的 aliyun 请求格式 + 响应解析 + 非 200 抛异常
- rerank_fn：候选池截断、token 预算丢尾、index 映回 + score 覆写、tail/leftover 保留、API 失败上抛
- hybrid_retriever 在 rerank_fn 抛异常时降级返回融合结果
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.rag.hybrid_retriever import retrieve
from core.rag.rerank import _DASHSCOPE_RERANK_URL, _call_rerank_api, build_rerank_fn
from core.rag.retrieval_config import RetrievalConfig


# ── fakes ────────────────────────────────────────────────────────────────────


def _fake_settings(
    *,
    api_key: str = "sk-test",
    enabled: bool = False,
    candidate_top_n: int = 20,
    max_request_tokens: int = 100_000,
    timeout_s: float = 5.0,
    model: str = "qwen3-rerank",
):
    emb = MagicMock()
    emb.api_key.get_secret_value.return_value = api_key
    rerank = SimpleNamespace(
        enabled=enabled,
        model=model,
        candidate_top_n=candidate_top_n,
        max_request_tokens=max_request_tokens,
        timeout_s=timeout_s,
    )
    return SimpleNamespace(embedding=emb, rerank=rerank)


def _doc(i: int, *, content: str | None = None) -> dict:
    return {
        "chunk_id": str(i),
        "content": content if content is not None else f"doc {i}",
        "score": float(100 - i),  # RRF 序：i 越小越靠前
        "file_path": f"/{i}.pdf",
    }


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """假装 httpx.AsyncClient：捕获请求，返回预设响应。"""

    def __init__(self, resp: _FakeResponse) -> None:
        self._resp = resp
        self.captured: tuple | None = None

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, json=None, headers=None) -> _FakeResponse:
        self.captured = (url, json, headers)
        return self._resp


# ── build_rerank_fn 惰性构造 ─────────────────────────────────────────────────


class TestBuildRerankFn:
    def test_none_when_no_api_key(self):
        with patch(
            "core.rag.rerank.get_settings",
            return_value=_fake_settings(api_key="", enabled=True),
        ):
            assert build_rerank_fn() is None
            assert build_rerank_fn(force=True) is None  # 无 key 即便 force 也 None

    def test_none_when_disabled_no_force(self):
        with patch(
            "core.rag.rerank.get_settings",
            return_value=_fake_settings(api_key="sk", enabled=False),
        ):
            assert build_rerank_fn() is None
            assert build_rerank_fn(force=True) is not None  # force 绕过开关

    def test_returns_fn_when_enabled(self):
        with patch(
            "core.rag.rerank.get_settings",
            return_value=_fake_settings(api_key="sk", enabled=True),
        ):
            fn = build_rerank_fn()
        assert callable(fn)


# ── _call_rerank_api 契约 ────────────────────────────────────────────────────


class TestCallRerankApi:
    async def test_aliyun_request_and_response(self):
        payload = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        }
        fake = _FakeClient(_FakeResponse(200, payload))
        with patch("core.rag.rerank.httpx.AsyncClient", return_value=fake):
            out = await _call_rerank_api(
                "q", ["d0", "d1"], model="qwen3-rerank", api_key="sk", top_n=2, timeout_s=5.0
            )
        url, body, headers = fake.captured
        assert url == _DASHSCOPE_RERANK_URL
        assert body["model"] == "qwen3-rerank"
        assert body["input"]["query"] == "q"
        assert body["input"]["documents"] == ["d0", "d1"]
        assert body["parameters"]["top_n"] == 2
        assert headers["Authorization"] == "Bearer sk"
        assert out == [
            {"index": 1, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.2},
        ]

    async def test_raises_on_non_200(self):
        fake = _FakeClient(_FakeResponse(400, {}, text="bad request"))
        with patch("core.rag.rerank.httpx.AsyncClient", return_value=fake):
            with pytest.raises(RuntimeError, match="400"):
                await _call_rerank_api(
                    "q", ["d0"], model="m", api_key="sk", top_n=1, timeout_s=5.0
                )

    async def test_malformed_results_returns_empty(self):
        fake = _FakeClient(_FakeResponse(200, {"output": {"results": "not-a-list"}}))
        with patch("core.rag.rerank.httpx.AsyncClient", return_value=fake):
            out = await _call_rerank_api(
                "q", ["d0"], model="m", api_key="sk", top_n=1, timeout_s=5.0
            )
        assert out == []


# ── rerank_fn 逻辑 ───────────────────────────────────────────────────────────


def _patched_fn(
    *,
    api_key="sk",
    candidate_top_n=20,
    max_request_tokens=100_000,
    api_returns=None,
):
    """构造一个已注入 fake settings + fake API 的 rerank_fn，并返回 (fn, captured)。"""

    captured: dict = {}

    async def fake_api(query, documents, **kw):
        captured["docs"] = documents
        captured["kw"] = kw
        if api_returns is not None:
            return api_returns
        return [{"index": i, "relevance_score": 0.1} for i in range(len(documents))]

    settings = _fake_settings(
        api_key=api_key, enabled=True, candidate_top_n=candidate_top_n,
        max_request_tokens=max_request_tokens,
    )
    patcher_s = patch("core.rag.rerank.get_settings", return_value=settings)
    patcher_a = patch("core.rag.rerank._call_rerank_api", new=fake_api)
    patcher_s.start()
    patcher_a.start()
    fn = build_rerank_fn()
    return fn, captured, (patcher_s, patcher_a)


class TestRerankFn:
    async def test_empty_docs_returns_unchanged(self):
        fn, captured, patchers = _patched_fn()
        try:
            out = await fn("query", [], top_n=5)
        finally:
            for p in patchers:
                p.stop()
        assert out == []
        assert "docs" not in captured  # 空入参不调 API

    async def test_candidate_pool_truncation(self):
        """40 条候选只送前 candidate_top_n=20 给 API，尾部不参与重排序。"""
        docs = [_doc(i) for i in range(40)]
        fn, captured, patchers = _patched_fn(candidate_top_n=20)
        try:
            out = await fn("query", docs, top_n=5)
        finally:
            for p in patchers:
                p.stop()
        assert len(captured["docs"]) == 20  # 只送 20
        assert len(out) == 40  # tail(20) 原样保留，不丢结果

    async def test_token_budget_drops_tail(self):
        """max_request_tokens 超限时丢弃尾部候选，减少送 API 的条数。"""
        # 每条约 3000 字符 ≈ 2000 token；预算 5000 → 只装得下 2 条
        docs = [_doc(i, content="x" * 3000) for i in range(20)]
        fn, captured, patchers = _patched_fn(max_request_tokens=5000)
        try:
            await fn("query", docs, top_n=5)
        finally:
            for p in patchers:
                p.stop()
        assert len(captured["docs"]) < 20  # 预算丢尾生效

    async def test_index_mapback_and_score_override(self):
        """API 结果按 index 映回原 dict，relevance_score 覆写 score（量纲统一）。"""
        docs = [
            {"chunk_id": "a", "content": "A", "score": 0.9, "file_path": "/1"},
            {"chunk_id": "b", "content": "B", "score": 0.5, "file_path": "/2"},
            {"chunk_id": "c", "content": "C", "score": 0.3, "file_path": "/3"},
        ]
        # API 判 idx 2 最相关，其次 idx 0；idx 1 被过滤（未返回）
        api_returns = [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.42},
        ]
        fn, _, patchers = _patched_fn(api_returns=api_returns)
        try:
            out = await fn("query", docs, top_n=5)
        finally:
            for p in patchers:
                p.stop()
        # reranked：c(0.95)、a(0.42)
        assert out[0]["chunk_id"] == "c"
        assert out[0]["score"] == 0.95
        assert out[1]["chunk_id"] == "a"
        assert out[1]["score"] == 0.42
        # leftover：b 未被打分，原序补后，保留原 score
        assert out[2]["chunk_id"] == "b"
        assert out[2]["score"] == 0.5
        assert len(out) == 3
        # 精排前后分数都保留（新增字段）；score 覆写行为不变
        assert out[0]["rerank_score"] == 0.95          # c
        assert out[0]["pre_rerank_score"] == 0.3       # c 入参原始 score（无 fused_score 时回退）
        assert out[0]["score"] == 0.95
        assert out[1]["rerank_score"] == 0.42          # a
        assert out[1]["pre_rerank_score"] == 0.9
        # leftover（b）未精排，不带新字段，score 保持原值
        assert "rerank_score" not in out[2]
        assert "pre_rerank_score" not in out[2]
        # 原入参不被改写（score 覆写在 dict copy 上）
        assert docs[0]["score"] == 0.9

    async def test_tail_preserved_when_pool_smaller_than_docs(self):
        """candidate_top_n < len(docs) 时，超出部分作 tail 原样保留。"""
        docs = [_doc(i) for i in range(4)]
        api_returns = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ]
        fn, captured, patchers = _patched_fn(candidate_top_n=2, api_returns=api_returns)
        try:
            out = await fn("query", docs, top_n=5)
        finally:
            for p in patchers:
                p.stop()
        assert len(captured["docs"]) == 2  # 只送 pool[:2]
        assert len(out) == 4  # reranked(2) + tail(2)
        assert [d["chunk_id"] for d in out] == ["0", "1", "2", "3"]
        # tail 保留原 score（未被覆写）
        assert out[2]["score"] == float(100 - 2)
        assert out[3]["score"] == float(100 - 3)

    async def test_api_failure_propagates(self):
        """rerank_fn 在 API 失败时上抛异常（由 hybrid_retriever 降级兜住）。"""
        docs = [_doc(i) for i in range(3)]
        settings = _fake_settings(api_key="sk", enabled=True)
        with (
            patch("core.rag.rerank.get_settings", return_value=settings),
            patch(
                "core.rag.rerank._call_rerank_api",
                new=AsyncMock(side_effect=RuntimeError("api down")),
            ),
        ):
            fn = build_rerank_fn()
            with pytest.raises(RuntimeError, match="api down"):
                await fn("query", docs, top_n=5)


# ── hybrid_retriever 降级集成 ────────────────────────────────────────────────


class TestHybridRetrieverDegrades:
    async def test_rerank_failure_returns_fused(self):
        """rerank_fn 抛异常时，hybrid_retriever 降级返回融合结果（未精排）。"""
        async def dense_fn(query, k):
            return [
                {"chunk_id": "1", "content": "a", "score": 0.5},
                {"chunk_id": "2", "content": "b", "score": 0.3},
            ]

        async def bad_rerank(query, docs, top_n):
            raise RuntimeError("api down")

        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=True)
        out = await retrieve(
            "q", "course1", cfg, dense_search_fn=dense_fn, rerank_fn=bad_rerank
        )
        # 融合结果原样返回（单路 dense，顺序不变），不被空异常吞成 []
        assert len(out) == 2
        assert out[0]["chunk_id"] == "1"


# ── hybrid_retriever 相关性阈值门控（第 2 层防线）──────────────────────────────


def _gate_docs(*ids: str) -> list[dict]:
    """造 dense 召回候选（带 RRF 量纲 score，不含 rerank_score）。"""
    return [
        {"chunk_id": cid, "content": f"c{cid}", "score": 1.0 / (60 + i + 1), "file_path": ""}
        for i, cid in enumerate(ids)
    ]


def _scored_rerank(score_map: dict[str, float]):
    """rerank_fn：给 score_map 中的 doc 写 rerank_score（降序在前），

    不在 score_map 的 doc 原样保留作 leftover（无 rerank_score）--模拟 rerank.py
    里 API 未返回的候选。用于验证阈值对「有分低分」与「无分 leftover」的处置。
    """

    async def rerank_fn(query, docs, top_n=5):
        scored, leftover = [], []
        for d in docs:
            if d["chunk_id"] in score_map:
                nd = dict(d)
                nd["rerank_score"] = score_map[d["chunk_id"]]
                scored.append(nd)
            else:
                leftover.append(dict(d))
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored + leftover

    return rerank_fn


class TestMinRerankScoreGate:
    """hybrid_retriever.retrieve 的 min_rerank_score 阈值门控。

    阈值只认 rerank_score（RRF/裸余弦/ts_rank 绝对值无相关性含义）。默认 0.0=不过滤，
    行为零变化；生效前置 rerank_enabled 且精排成功拿到 rerank_score。
    """

    async def test_default_zero_no_filter(self):
        """min_rerank_score=0.0（默认）-> 不过滤，含无分 leftover 全保留（行为零变化）。"""
        async def dense_fn(query, k):
            return _gate_docs("1", "2", "3", "4")

        rerank_fn = _scored_rerank({"1": 0.9, "2": 0.2, "3": 0.1})  # "4" 未打分=leftover
        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=True)  # min_rerank_score 默认 0.0
        out = await retrieve("q", "c1", cfg, dense_search_fn=dense_fn, rerank_fn=rerank_fn)

        assert len(out) == 4  # 3 打分 + 1 leftover，全保留
        assert {d["chunk_id"] for d in out} == {"1", "2", "3", "4"}

    async def test_threshold_filters_low_and_leftover(self):
        """阈值>0 -> 丢低分 doc + 丢无分 leftover（只认 rerank_score）。"""
        async def dense_fn(query, k):
            return _gate_docs("1", "2", "3", "4")

        rerank_fn = _scored_rerank({"1": 0.9, "2": 0.2, "3": 0.6})  # "4" 未打分=leftover
        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=True, min_rerank_score=0.5)
        out = await retrieve("q", "c1", cfg, dense_search_fn=dense_fn, rerank_fn=rerank_fn)

        assert {d["chunk_id"] for d in out} == {"1", "3"}  # 0.9、0.6 留；0.2、leftover(4) 丢

    async def test_threshold_all_filtered_returns_empty(self):
        """全部低于阈值 -> 返回 []（下游 retrieve_context 见空返 "" -> _execute_rag 拒答，见 test_rag_tool_signal）。"""
        async def dense_fn(query, k):
            return _gate_docs("1", "2")

        rerank_fn = _scored_rerank({"1": 0.1, "2": 0.2})
        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=True, min_rerank_score=0.5)
        out = await retrieve("q", "c1", cfg, dense_search_fn=dense_fn, rerank_fn=rerank_fn)

        assert out == []

    async def test_threshold_no_effect_when_rerank_disabled(self):
        """rerank_enabled=False + 阈值>0 -> 不过滤、不调 rerank、RRF 分原样保留（不误杀 RRF 分）。"""
        async def dense_fn(query, k):
            return _gate_docs("1", "2", "3")

        calls: list[int] = []

        async def rerank_fn(query, docs, top_n=5):
            calls.append(1)
            return docs  # 不该被调

        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=False, min_rerank_score=0.5)
        out = await retrieve("q", "c1", cfg, dense_search_fn=dense_fn, rerank_fn=rerank_fn)

        assert calls == []  # rerank 未被调用
        assert len(out) == 3  # RRF 融合结果原样返回，无过滤
        assert {d["chunk_id"] for d in out} == {"1", "2", "3"}

    async def test_threshold_no_effect_on_rerank_failure(self):
        """精排 API 抛异常 + 阈值>0 -> 降级返回融合结果且不被清空（阈值在 try 内，降级时不生效）。"""
        async def dense_fn(query, k):
            return _gate_docs("1", "2")

        async def bad_rerank(query, docs, top_n):
            raise RuntimeError("api down")

        cfg = RetrievalConfig(bm25_enabled=False, rerank_enabled=True, min_rerank_score=0.5)
        out = await retrieve("q", "c1", cfg, dense_search_fn=dense_fn, rerank_fn=bad_rerank)

        assert len(out) == 2  # 降级到融合结果，未被阈值清空
        assert {d["chunk_id"] for d in out} == {"1", "2"}
