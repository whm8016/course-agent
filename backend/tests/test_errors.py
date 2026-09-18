"""统一错误信封（core/observability/errors.py）规格测试。

错误处理与可观测性重整 Step 1：错误分类注册表 + 信封字段 + ref 来源优先级。
spec 先行——断言的是设计约定（错误码封闭集、字段名、截断语义），非对实现回填。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from core.observability.context import bind_context
from core.observability.errors import (
    ERROR_CODES,
    ERROR_REGISTRY,
    classify,
    emit_error_event,
    error_ref,
    normalize_error,
)


# ---------------------------------------------------------------------------
# 注册表封闭集（8 码）与文案非空
# ---------------------------------------------------------------------------

EXPECTED_CODES = {
    "llm_unavailable", "context_overflow", "schema_invalid", "tool_failed",
    "capability_failed", "cancelled", "unauthorized", "internal",
}


def test_registry_is_closed_set_of_eight_codes():
    assert ERROR_CODES == EXPECTED_CODES
    assert set(ERROR_REGISTRY) == EXPECTED_CODES
    for code, spec in ERROR_REGISTRY.items():
        assert spec.code == code
        assert spec.user_message.strip(), f"{code} 缺用户文案"


# ---------------------------------------------------------------------------
# classify 分类表
# ---------------------------------------------------------------------------

_REQ = httpx.Request("POST", "http://test.local/v1/chat/completions")


@pytest.mark.parametrize(
    ("exc_factory", "expected_code"),
    [
        pytest.param(lambda: asyncio.CancelledError(), "cancelled", id="CancelledError"),
        pytest.param(
            lambda: __import__("core.llm.reliability", fromlist=["CircuitOpenError"]).CircuitOpenError("OPEN"),
            "llm_unavailable", id="CircuitOpenError",
        ),
        pytest.param(
            lambda: __import__("core.llm.reliability", fromlist=["LLMRetryError"]).LLMRetryError("exhausted"),
            "llm_unavailable", id="LLMRetryError",
        ),
        pytest.param(lambda: RuntimeError("This model's maximum context length is 4096 tokens"),
                     "context_overflow", id="keyword:maximum-context"),
        pytest.param(
            lambda: httpx.HTTPStatusError("too large", request=_REQ,
                                          response=httpx.Response(413, request=_REQ)),
            "context_overflow", id="status:413",
        ),
        pytest.param(lambda: json.JSONDecodeError("Expecting value", "doc", 0),
                     "schema_invalid", id="JSONDecodeError"),
        pytest.param(lambda: RuntimeError("503 Service Unavailable"),
                     "llm_unavailable", id="keyword:provider-503"),
        pytest.param(lambda: RuntimeError("model does not support image input"),
                     "internal", id="unknown-biz-error"),
        pytest.param(lambda: ValueError("bad arg"),
                     "internal", id="ValueError"),
    ],
)
def test_classify_table(exc_factory, expected_code):
    assert classify(exc_factory()).code == expected_code


def test_llm_unavailable_message_reuses_reliability_constant():
    """文案单一真相源：reliability.LLM_SERVICE_UNAVAILABLE 引用本表（不双写）。"""
    from core.llm.reliability import LLM_SERVICE_UNAVAILABLE

    assert ERROR_REGISTRY["llm_unavailable"].user_message == LLM_SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# error_ref：turn_id 优先 / request_id 回退 / 取前 8 位
# ---------------------------------------------------------------------------


def test_error_ref_prefers_turn_id():
    bind_context(turn_id="turn_abcdefgh1234", request_id="req_zzzzzzzz")
    assert error_ref() == "turn_abc"  # 前 8 位："turn_abc"


def test_error_ref_falls_back_to_request_id():
    bind_context(turn_id="", request_id="req_abcdef123456")
    assert error_ref() == "req_abcd"  # 前 8 位："req_abcd"


def test_error_ref_empty_when_no_context():
    bind_context(turn_id="", request_id="")
    assert error_ref() == ""


# ---------------------------------------------------------------------------
# normalize_error：信封五元组
# ---------------------------------------------------------------------------


def test_normalize_error_envelope_fields():
    exc = RuntimeError("x" * 500)
    env = normalize_error(exc, message_override="深度解题失败，请稍后重试。")
    assert env["code"] == "internal"
    assert env["message"] == "深度解题失败，请稍后重试。"
    assert env["exc_type"] == "RuntimeError"
    assert len(env["exc_head"]) == 200  # 200 字截断
    assert set(env) == {"code", "message", "ref", "exc_type", "exc_head"}


def test_normalize_error_default_message_from_registry():
    env = normalize_error(asyncio.CancelledError())
    assert env["code"] == "cancelled"
    assert env["message"] == ERROR_REGISTRY["cancelled"].user_message


# ---------------------------------------------------------------------------
# emit_error_event：bus 收到完整信封、异常现场记日志、返回信封
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_error_event_sends_envelope_to_bus():
    bus = MagicMock()
    bus.error = AsyncMock()
    bind_context(turn_id="turn_t123456789")

    exc = RuntimeError("upstream exploded")
    env = await emit_error_event(bus, exc=exc, source="orchestrator")

    assert env["code"] == "internal"
    bus.error.assert_awaited_once()
    call = bus.error.await_args
    assert call.args[0] == ERROR_REGISTRY["internal"].user_message  # message 位置参数
    kwargs = call.kwargs
    assert kwargs["source"] == "orchestrator"
    assert kwargs["code"] == "internal"
    assert kwargs["ref"] == "turn_t12"  # "turn_t123456789"[:8]
    assert kwargs["exc_type"] == "RuntimeError"
    assert kwargs["exc_head"] == "upstream exploded"
