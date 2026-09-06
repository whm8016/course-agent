"""MCP 主动发现对比 demo 回归测试（离线，不调真实 LLM）。

覆盖：
- 合成池契约（80 个 / wrapped 名唯一 / deferred / schema 形状）
- 任务 expected_tools 全部落在池内（打分标准可满足）
- manifest 渲染包含全部工具名（arm A 的"全量索引"前提）
- LocalEmbedder 分词与 ToolIndex 语义检索（canonical need → 期望工具 top 命中）
- DiscoverToolLoader：live list 追加 / 重复 discover 去重 / query 记录
- 两臂 harness 接线（loop_fn 注入 fake loop，fake loop 内部真调 execute_tool，
  验证 contextvar 注入 + registry 路由 + live list 增长）
- 指标计算（recall / false_calls / missed / load_tools_used / discover_calls）
- fail-fast（expected 不在池内抛 ValueError）与 summarize 聚合
"""
from __future__ import annotations

import pytest

from core.agentic.types import LoopOutcome
from core.observability.cost import TokenUsage


@pytest.fixture(scope="module", autouse=True)
def _clean_env():
    """模块级启用评测环境（registry 注册 + 全局重定向），跑完整体还原。

    突变/还原清单同源在 harness.ensure_setup 的还原闭包里——这里不做镜像维护；
    不还原的话，本模块之后跑的用例（如显式指定文件顺序的 test_deferred_tools）
    会把 loaded_tools.json 静默写进评测目录、token 统计类测试读到被关掉的开关。
    """
    from scripts.eval_mcp_discovery.harness import ensure_setup

    restore = ensure_setup()
    yield
    restore()


@pytest.fixture()
def sessions_tmp(tmp_path, monkeypatch):
    """把 session 持久化目录指向临时目录（load_tools 的 _persist 不落 results/）。"""
    import core.mcp.session_state as ss

    monkeypatch.setattr(ss, "_SESSIONS_BASE", tmp_path)
    return tmp_path


# ── 合成池与任务契约 ─────────────────────────────────────────────────────────


def test_pool_build_contract():
    from scripts.eval_mcp_discovery import synthetic_tools

    pool = synthetic_tools.build_pool()
    assert len(pool) == 80
    names = [a.wrapped_name for a in pool]
    assert len(names) == len(set(names)), "wrapped 名必须唯一"
    assert all(a.deferred for a in pool), "对比前提：全部按 deferred 揭示"
    assert all(a.wrapped_name.startswith("mcp_") for a in pool)
    schema = pool[0].to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == pool[0].wrapped_name
    # 至少 10 个虚构 server（跨域覆盖）
    assert len({a.server_name for a in pool}) >= 10


def test_tasks_expected_tools_in_pool():
    from scripts.eval_mcp_discovery import synthetic_tools
    from scripts.eval_mcp_discovery.tasks import get_tasks

    pool_names = {a.wrapped_name for a in synthetic_tools.build_pool()}
    tasks = get_tasks()
    assert len(tasks) == 6
    for t in tasks:
        assert 2 <= len(t["expected_tools"]) <= 4, f"{t['id']} 每题 2-4 个工具"
        missing = [e for e in t["expected_tools"] if e not in pool_names]
        assert not missing, f"{t['id']} expected 不在池内: {missing}"
        assert t["prompt"].strip()


def test_manifest_renders_all_tools():
    from core.mcp.deferred_tools import render_deferred_tools_manifest
    from scripts.eval_mcp_discovery import synthetic_tools

    pool = synthetic_tools.build_pool()
    manifest = render_deferred_tools_manifest(pool)
    assert "## 扩展工具" in manifest
    for a in pool:
        assert a.wrapped_name in manifest


# ── 嵌入与语义检索 ───────────────────────────────────────────────────────────


def test_local_embedder_tokens():
    from scripts.eval_mcp_discovery.discovery import _tokens

    toks = _tokens("查询 USD 汇率")
    assert "usd" in toks  # 英文按词
    assert "汇" in toks and "汇率" in toks  # 中文 unigram + bigram


def test_tool_index_canonical_queries():
    from scripts.eval_mcp_discovery import synthetic_tools
    from scripts.eval_mcp_discovery.discovery import ToolIndex

    index = ToolIndex(synthetic_tools.build_pool())
    cases = {
        "查询两个国家货币之间的实时汇率": "mcp_forex_get_rate",
        "按关键词检索学术论文": "mcp_arxiv_search",
        "查一下东京未来几天的天气预报": "mcp_weather_forecast",
        "把一段英文文本翻译成中文": "mcp_translate_text",
        "根据数据画一张柱状图": "mcp_chart_render_chart",
        "把文件压缩成 zip 包": "mcp_file_compress",
        "查比特币的实时价格": "mcp_crypto_price",
    }
    for need, expect_top1 in cases.items():
        hits = index.search(need, top_k=3)
        assert hits, f"检索无结果: {need}"
        assert hits[0][0].wrapped_name == expect_top1, f"{need} 期望 {expect_top1}，实得 {hits[0][0].wrapped_name}"


def test_discover_loader_appends_and_dedups():
    from scripts.eval_mcp_discovery import synthetic_tools
    from scripts.eval_mcp_discovery.discovery import DiscoverToolLoader

    live: list[dict] = [{"type": "function", "function": {"name": "discover_tools"}}]
    loader = DiscoverToolLoader(pool=synthetic_tools.build_pool(), top_k=3)
    loader.bind_live_schemas(live)
    assert loader.initial_schemas() == []

    out1 = loader.discover("查询两个国家货币之间的实时汇率")
    assert out1["loaded"], "首次 discover 应有命中"
    appended = {s["function"]["name"] for s in live[1:]}
    assert appended == {t["name"] for t in out1["loaded"]}
    assert "mcp_forex_get_rate" in appended

    # 同一 need 再次 discover：排序确定（哈希词袋），top-k 完全相同 → 全部去重，
    # 不再重复 append，走 already_loaded
    before = len(live)
    out2 = loader.discover("查询两个国家货币之间的实时汇率")
    assert not out2["loaded"] and out2["already_loaded"]
    assert len(live) == before
    assert loader.discover_queries == ["查询两个国家货币之间的实时汇率"] * 2


def test_discover_loader_empty_need_returns_nothing():
    from scripts.eval_mcp_discovery import synthetic_tools
    from scripts.eval_mcp_discovery.discovery import DiscoverToolLoader

    loader = DiscoverToolLoader(pool=synthetic_tools.build_pool()[:5], top_k=3)
    loader.bind_live_schemas([])
    assert loader.discover("   ") == {"loaded": [], "already_loaded": []}


# ── harness 两臂接线（loop_fn 注入，内部真调 execute_tool）──────────────────


def _outcome(tools_used: list[str], rounds: int = 3) -> LoopOutcome:
    return LoopOutcome(
        final_text="任务完成：这是结果总结。",
        rounds=rounds,
        tools_used=tools_used,
        completed=True,
        total_usage=TokenUsage(input_tokens=1234, output_tokens=567, cache_read_tokens=100),
    )


@pytest.mark.asyncio
async def test_arm_deferred_exact_wiring(sessions_tmp):
    """arm A：fake loop 内真调 load_tools + mock 工具，验证生产 contextvar 路由。"""
    from scripts.eval_mcp_discovery.harness import run_case
    from core.agent.tool_registry import execute_tool

    task = {"id": "paper_translate", "prompt": "查论文并翻译摘要",
            "expected_tools": ["mcp_arxiv_search", "mcp_translate_text"]}

    async def fake_loop(**kwargs):
        schemas = kwargs["tool_schemas"]
        assert schemas[0]["function"]["name"] == "load_tools", "arm A 初始应挂 load_tools"
        assert "扩展工具" in kwargs["system_prompt"], "arm A 系统提示词应含全量 manifest"

        res = await execute_tool("load_tools", names=["mcp_arxiv_search", "mcp_translate_text"])
        assert res.success, res.content
        names_in_live = {s["function"]["name"] for s in schemas}
        assert {"mcp_arxiv_search", "mcp_translate_text"} <= names_in_live

        r = await execute_tool("mcp_arxiv_search", course_id="general", user_id="eval", query="transformer")
        assert r.success and "[mock] mcp_arxiv_search" in r.content
        return _outcome(["load_tools", "mcp_arxiv_search", "mcp_translate_text"])

    rec = await run_case(task, "deferred_exact", loop_fn=fake_loop, run_tag="t")

    assert rec["error"] is None
    assert rec["load_tools_used"] is True
    assert rec["recall"] == 1.0
    assert rec["false_calls"] == []
    assert rec["guessed_calls"] == []  # 经 load_tools 揭示后调用，非猜名直调
    assert rec["input_tokens"] == 1234 and rec["output_tokens"] == 567 and rec["cache_read_tokens"] == 100
    assert rec["system_prompt_chars"] > 2000, "80 工具 manifest 应显著长于 discover 引导块"
    # 持久化落在临时目录，未污染真实 data/sessions/
    assert list(sessions_tmp.rglob("loaded_tools.json"))


@pytest.mark.asyncio
async def test_arm_active_discovery_wiring(sessions_tmp):
    """arm B：fake loop 内真调 discover_tools，验证独立 contextvar + live list 增长。"""
    from scripts.eval_mcp_discovery.harness import run_case
    from core.agent.tool_registry import execute_tool

    task = {"id": "paper_translate", "prompt": "查论文并翻译摘要",
            "expected_tools": ["mcp_arxiv_search", "mcp_translate_text"]}

    async def fake_loop(**kwargs):
        schemas = kwargs["tool_schemas"]
        assert schemas[0]["function"]["name"] == "discover_tools", "arm B 初始只挂 discover_tools"
        assert "扩展工具" not in kwargs["system_prompt"], "arm B 系统提示词不应含工具清单"
        assert "工具发现" in kwargs["system_prompt"]

        r1 = await execute_tool("discover_tools", need="检索学术论文查 transformer 论文")
        assert r1.success, r1.content
        r2 = await execute_tool("discover_tools", need="把英文文本翻译成中文")
        assert r2.success, r2.content
        live_names = {s["function"]["name"] for s in schemas}
        assert "mcp_arxiv_search" in live_names
        assert "mcp_translate_text" in live_names

        r3 = await execute_tool("mcp_translate_text", text="attention is all you need", source="en", target="zh")
        assert r3.success
        # 故意加一个未 discover 过的猜名直调，验证 guessed_calls 能把它挑出来
        return _outcome(["discover_tools", "discover_tools", "mcp_arxiv_search",
                         "mcp_translate_text", "mcp_weather_current"])

    rec = await run_case(task, "active_discovery", loop_fn=fake_loop, run_tag="t")

    assert rec["error"] is None
    assert rec["recall"] == 1.0
    assert rec["false_calls"] == ["mcp_weather_current"]
    assert rec["guessed_calls"] == ["mcp_weather_current"]  # 未 discover 直调，与 false_calls 区分口径
    assert rec["discover_calls"] == 2
    assert rec["load_tools_used"] is False
    # discover 引导块远小于 80 工具 manifest（arm B 的 prompt 成本优势）
    assert rec["system_prompt_chars"] < 500


@pytest.mark.asyncio
async def test_metrics_partial_recall_and_false_calls(sessions_tmp):
    """召回不满分 + 误召：expected 3 个中 2 个，多调 1 个池内无关工具。"""
    from scripts.eval_mcp_discovery.harness import run_case

    task = {"id": "ai_edu_news", "prompt": "新闻情绪图表",
            "expected_tools": ["mcp_news_search", "mcp_news_sentiment_analysis", "mcp_chart_render_chart"]}

    async def fake_loop(**kwargs):
        return _outcome(["mcp_news_search", "mcp_chart_render_chart", "mcp_weather_current"])

    rec = await run_case(task, "active_discovery", loop_fn=fake_loop, run_tag="t")
    assert rec["error"] is None
    assert rec["recall"] == round(2 / 3, 4)
    assert rec["missed_tools"] == ["mcp_news_sentiment_analysis"]
    assert rec["false_calls"] == ["mcp_weather_current"]


@pytest.mark.asyncio
async def test_run_case_error_tolerated(sessions_tmp):
    """loop 抛错不炸 harness：rec 带 error，其余字段缺省。"""
    from scripts.eval_mcp_discovery.harness import run_case

    async def boom(**kwargs):
        raise RuntimeError("LLM 挂了")

    task = {"id": "tokyo_trip", "prompt": "x", "expected_tools": ["mcp_weather_forecast"]}
    rec = await run_case(task, "deferred_exact", loop_fn=boom, run_tag="t")
    assert rec["error"] is not None and "LLM 挂了" in rec["error"]
    assert rec.get("rounds") is None  # metrics 未采集


@pytest.mark.asyncio
async def test_run_case_failfast_on_missing_expected(sessions_tmp):
    from scripts.eval_mcp_discovery.harness import run_case

    task = {"id": "bad", "prompt": "x", "expected_tools": ["mcp_nonexistent_tool"]}

    async def never(**kwargs):  # pragma: no cover
        raise AssertionError("不应触达 loop")

    with pytest.raises(ValueError, match="不在合成池内"):
        await run_case(task, "deferred_exact", loop_fn=never)


# ── 汇总聚合 ─────────────────────────────────────────────────────────────────


def test_summarize_by_arm():
    from scripts.eval_mcp_discovery.run_eval import summarize

    recs = [
        {"arm": "a", "error": None, "input_tokens": 100, "output_tokens": 50, "rounds": 2,
         "wall_ms": 1000, "recall": 1.0, "false_calls": [], "system_prompt_chars": 10},
        {"arm": "a", "error": "boom"},
        {"arm": "b", "error": None, "input_tokens": 300, "output_tokens": 60, "rounds": 4,
         "wall_ms": 3000, "recall": 0.5, "false_calls": ["x"], "system_prompt_chars": 20},
    ]
    s = summarize(recs)
    assert s["a"]["cases"] == 2 and s["a"]["errors"] == 1
    assert s["a"]["avg_input_tokens"] == 100  # error 行不进均值
    assert s["b"]["avg_recall"] == 0.5 and s["b"]["avg_false_calls"] == 1

    # 全 error 臂：avg 字段为 None（无数据），不得以 0 的形态出现
    s2 = summarize([{"arm": "x", "error": "boom"}])
    assert s2["x"] == {"cases": 1, "errors": 1}
    assert "avg_recall" not in s2["x"]
