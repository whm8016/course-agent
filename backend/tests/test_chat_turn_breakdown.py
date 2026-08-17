"""chat.turn_breakdown 汇总日志单测 —— 首字延迟分步计时（只计时，不改行为）。

mock 掉 DB/LLM/skill/MCP 后跑完整 ChatPipeline.run，从 caplog 抓 chat.turn_breakdown：
断言字段齐全、毫秒字段为非负整数（不测真实耗时数值）。覆盖：
  1. 正常回合（首轮即作答）→ 全字段 + extra_rounds=0
  2. 无 user_id（bot/无登录路径）→ stitch_ms/mem0_ms 缺省 0，仍打日志
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context import UnifiedContext
from core.stream_bus import StreamBus


def _mock_skill_service():
    """假 SkillService：无 always-on skill 全文、无 manifest 条目（纯 MagicMock 会炸 join）。"""
    svc = MagicMock()
    svc.load_always_for_context.return_value = ""
    svc.summary_entries.return_value = []
    return svc

# breakdown 里恒为非负整数的毫秒字段
_MS_FIELDS = (
    "profile_ms", "quota_ms", "stitch_ms", "mem0_ms", "prompt_ms",
    "kb_seed_ms", "image_ms", "pre_llm_ms", "llm0_total_ms", "elapsed_ms",
)


# LogRecord 自带的标准属性（从 r.__dict__ 里剔除，只留 log_flow 的 extra 字段）
_LOG_STD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})


def _breakdown_records(caplog) -> list[dict[str, Any]]:
    """log_flow 走 logging extra=，字段挂在 LogRecord 属性上，从 __dict__ 提取。"""
    return [
        {k: v for k, v in r.__dict__.items() if k not in _LOG_STD_ATTRS}
        for r in caplog.records
        if getattr(r, "stage", None) == "chat.turn_breakdown"
    ]


class _FakeRows:
    """假 db.execute 结果（MCP enrollment 查询 mock 用）。"""

    def __init__(self, rows: list = None):
        self._rows = rows or []

    def all(self):
        return self._rows


class _FakeSessionFactory:
    """假 AsyncSessionLocal（session factory）：调用返回可 async-with 的假 session。"""

    def __init__(self, rows: list | None = None):
        self._rows = _FakeRows(rows)

    def __call__(self):
        return _FakeSession(self._rows)


class _FakeSession:
    """假 AsyncSession 上下文管理器（stitch/MCP 查询共用，无真实 DB）。"""

    def __init__(self, rows: _FakeRows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, *a, **kw):
        return self._rows


@pytest.mark.asyncio
async def test_turn_breakdown_fields_complete(caplog):
    """正常回合：breakdown 字段齐全，毫秒非负整数，extra_rounds=0。"""
    from core.capabilities.chat_pipeline import ChatPipeline

    caplog.set_level(logging.INFO, logger="flow")
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="什么是基尔霍夫电压定律？",
        enabled_tools=["rag"],
    )
    bus = StreamBus()
    outcome = MagicMock(rounds=1, tools_used=[], first_ttft_ms=123, first_round_ms=456)
    with patch("core.capabilities.chat_pipeline.resolve_profile_runtime",
               AsyncMock(return_value=MagicMock(client=None, text_model=None, binding=None))), \
         patch("core.llm.prompts.get_course_prompt", AsyncMock(return_value="")), \
         patch("core.db.database.AsyncSessionLocal", _FakeSessionFactory()), \
         patch("core.memory.proactive.stitch_for_turn",
               AsyncMock(return_value=MagicMock(text="", matched_topic=None,
                                                evidence_topic=None, should_stitch=False))), \
         patch("core.memory.mem0_client.build_memory_context", AsyncMock(return_value="")), \
         patch("core.prompt_loader.load_prompt_dict",
               return_value={"loop": {"system": ""}, "kb_seed": {"header": "[KB]"}}), \
         patch("core.skills.skill_service.get_skill_service", return_value=_mock_skill_service()), \
         patch("core.capabilities.chat_pipeline.retrieve_kb_seed", AsyncMock(return_value="")), \
         patch("core.capabilities.chat_pipeline.describe_images", AsyncMock(side_effect=lambda c, b, r: b)), \
         patch("core.capabilities.chat_pipeline.run_agent_loop",
               AsyncMock(return_value=outcome)) as run_loop:
        await ChatPipeline().run(ctx, bus)

    run_loop.assert_awaited_once()
    records = _breakdown_records(caplog)
    assert len(records) == 1
    rec = records[0]
    for f in _MS_FIELDS:
        assert isinstance(rec[f], int) and rec[f] >= 0, f"{f}={rec[f]!r}"
    assert rec["llm0_ttft_ms"] == 123
    assert rec["llm0_total_ms"] == 456
    assert rec["extra_rounds"] == 0
    assert rec["tools_used"] == []
    assert rec["kb_on"] is False
    assert rec["pre_llm_ms"] == (
        rec["profile_ms"] + rec["quota_ms"] + rec["prompt_ms"]
        + rec["kb_seed_ms"] + rec["image_ms"]
    )
    # system_prompt_chars 存在且为非负整数
    assert isinstance(rec["system_prompt_chars"], int) and rec["system_prompt_chars"] >= 0
    if not bus._closed:
        await bus.close()


@pytest.mark.asyncio
async def test_turn_breakdown_multi_round_and_kb_on(caplog):
    """多一轮工具 + KB 命中：extra_rounds=1、kb_on=True、tools_used 透传。"""
    from core.capabilities.chat_pipeline import ChatPipeline

    caplog.set_level(logging.INFO, logger="flow")
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="帮我查示波器参数",
        enabled_tools=["rag"],
    )
    bus = StreamBus()
    outcome = MagicMock(rounds=2, tools_used=["rag"], first_ttft_ms=None, first_round_ms=789)

    async def _kb_seed(*a, **kw):
        return "[知识库预检索]\n证据"

    with patch("core.capabilities.chat_pipeline.resolve_profile_runtime",
               AsyncMock(return_value=MagicMock(client=None, text_model=None, binding=None))), \
         patch("core.llm.prompts.get_course_prompt", AsyncMock(return_value="")), \
         patch("core.db.database.AsyncSessionLocal", _FakeSessionFactory()), \
         patch("core.memory.proactive.stitch_for_turn",
               AsyncMock(return_value=MagicMock(text="", matched_topic=None,
                                                evidence_topic=None, should_stitch=False))), \
         patch("core.memory.mem0_client.build_memory_context", AsyncMock(return_value="")), \
         patch("core.prompt_loader.load_prompt_dict",
               return_value={"loop": {"system": ""}, "kb_seed": {"header": "[KB]"}}), \
         patch("core.skills.skill_service.get_skill_service", return_value=_mock_skill_service()), \
         patch("core.capabilities.chat_pipeline.retrieve_kb_seed", AsyncMock(side_effect=_kb_seed)), \
         patch("core.capabilities.chat_pipeline.describe_images", AsyncMock(side_effect=lambda c, b, r: b)), \
         patch("core.capabilities.chat_pipeline.run_agent_loop",
               AsyncMock(return_value=outcome)):
        await ChatPipeline().run(ctx, bus)

    records = _breakdown_records(caplog)
    assert len(records) == 1
    rec = records[0]
    assert rec["extra_rounds"] == 1
    assert rec["kb_on"] is True
    assert rec["tools_used"] == ["rag"]
    # provider 未吐 delta → first_ttft_ms=None，log_flow 跳过 None 字段 → 日志中缺席
    assert "llm0_ttft_ms" not in rec
    assert rec["llm0_total_ms"] == 789
    if not bus._closed:
        await bus.close()


@pytest.mark.asyncio
async def test_turn_breakdown_not_logged_when_loop_raises(caplog):
    """loop 抛错（首轮 LLM 失败上抛）→ 不打 breakdown（异常路径归 turn_runtime 错误链路）。"""
    from core.capabilities.chat_pipeline import ChatPipeline

    caplog.set_level(logging.INFO, logger="flow")
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="你好",
        enabled_tools=[],
    )
    bus = StreamBus()
    with patch("core.capabilities.chat_pipeline.resolve_profile_runtime",
               AsyncMock(return_value=MagicMock(client=None, text_model=None, binding=None))), \
         patch("core.llm.prompts.get_course_prompt", AsyncMock(return_value="")), \
         patch("core.db.database.AsyncSessionLocal", _FakeSessionFactory()), \
         patch("core.memory.proactive.stitch_for_turn",
               AsyncMock(return_value=MagicMock(text="", matched_topic=None,
                                                evidence_topic=None, should_stitch=False))), \
         patch("core.memory.mem0_client.build_memory_context", AsyncMock(return_value="")), \
         patch("core.prompt_loader.load_prompt_dict",
               return_value={"loop": {"system": ""}, "kb_seed": {"header": "[KB]"}}), \
         patch("core.skills.skill_service.get_skill_service", return_value=_mock_skill_service()), \
         patch("core.capabilities.chat_pipeline.retrieve_kb_seed", AsyncMock(return_value="")), \
         patch("core.capabilities.chat_pipeline.describe_images", AsyncMock(side_effect=lambda c, b, r: b)), \
         patch("core.capabilities.chat_pipeline.run_agent_loop",
               AsyncMock(side_effect=RuntimeError("LLM down"))):
        with pytest.raises(RuntimeError):
            await ChatPipeline().run(ctx, bus)

    assert _breakdown_records(caplog) == []
    if not bus._closed:
        await bus.close()


@pytest.mark.asyncio
async def test_timing_survives_malicious_preset_metadata(caplog):
    """WS 入口已剥离客户端下划线键（api/run.py::_sanitize_client_metadata），但
    bot/内网调用方仍可预置 metadata——_timing 预置非 dict（如字符串）不该让计时
    写入 TypeError 崩掉整轮，覆盖式合并直接顶掉非法值。"""
    from core.pipeline_common import build_common_context_layers

    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="欧姆定律",
        metadata={"_timing": "attacker-controlled-string"},
    )
    with patch("core.llm.prompts.get_course_prompt", AsyncMock(return_value="")), \
         patch("core.db.database.AsyncSessionLocal", _FakeSessionFactory()), \
         patch("core.memory.proactive.stitch_for_turn",
               AsyncMock(return_value=MagicMock(text="", matched_topic=None,
                                                evidence_topic=None, should_stitch=False))), \
         patch("core.memory.mem0_client.build_memory_context", AsyncMock(return_value="")):
        layers = await build_common_context_layers(ctx)

    assert isinstance(layers, object)  # 未抛异常即通过
    t = ctx.metadata["_timing"]
    assert isinstance(t, dict) and isinstance(t.get("stitch_ms"), int) and t["stitch_ms"] >= 0
    assert isinstance(t.get("mem0_ms"), int) and t["mem0_ms"] >= 0
