"""Tests for core.observability — context injection and log_flow format."""
from __future__ import annotations

import logging

import pytest

from core.observability.context import bind_context, get_context_fields
from core.observability.flow import log_flow
from core.observability.logging import ContextFilter


# ---------------------------------------------------------------------------
# context.py
# ---------------------------------------------------------------------------

def test_bind_and_get_context_fields():
    bind_context(turn_id="t1", user_id="u1", course_id="c1", mode="chat")
    fields = get_context_fields()
    assert fields["turn_id"] == "t1"
    assert fields["user_id"] == "u1"
    assert fields["course_id"] == "c1"
    assert fields["mode"] == "chat"


def test_bind_context_partial_update():
    bind_context(turn_id="t2")
    bind_context(course_id="c2")
    fields = get_context_fields()
    assert fields["turn_id"] == "t2"
    assert fields["course_id"] == "c2"


def test_bind_context_empty_value_excluded():
    bind_context(turn_id="", job_id="")
    fields = get_context_fields()
    assert "turn_id" not in fields or fields.get("turn_id", "") == ""


# ---------------------------------------------------------------------------
# logging.py — ContextFilter
# ---------------------------------------------------------------------------

class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_context_filter_injects_fields():
    bind_context(turn_id="inject-test", user_id="u99")
    cf = ContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    cf.filter(record)
    assert record.turn_id == "inject-test"  # type: ignore[attr-defined]
    assert record.user_id == "u99"  # type: ignore[attr-defined]


def test_context_filter_does_not_overwrite_explicit_fields():
    bind_context(turn_id="from-context")
    cf = ContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    record.turn_id = "explicit-value"  # type: ignore[attr-defined]
    cf.filter(record)
    # explicit value must be preserved
    assert record.turn_id == "explicit-value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# flow.py — log_flow
# ---------------------------------------------------------------------------

def test_log_flow_emits_stage_field(caplog):
    with caplog.at_level(logging.INFO, logger="flow"):
        log_flow("test.stage", iteration=3, elapsed_ms=100)
    assert any("test.stage" in r.message or getattr(r, "stage", "") == "test.stage"
               for r in caplog.records)


def test_log_flow_numeric_fields_pass_through():
    handler = _CapturingHandler()
    lg = logging.getLogger("flow.test.numeric")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

    log_flow("tool.result", logger=lg, elapsed_ms=250, result_chars=1024, status="ok")

    assert len(handler.records) == 1
    rec = handler.records[0]
    assert rec.elapsed_ms == 250  # type: ignore[attr-defined]
    assert rec.result_chars == 1024  # type: ignore[attr-defined]
    assert rec.status == "ok"  # type: ignore[attr-defined]


def test_log_flow_string_fields_clipped():
    handler = _CapturingHandler()
    lg = logging.getLogger("flow.test.clip")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

    long_str = "a" * 800
    log_flow("chat.start", logger=lg, question=long_str)

    rec = handler.records[0]
    # string fields become _chars + _head
    assert rec.question_chars == 800  # type: ignore[attr-defined]
    head = rec.question_head  # type: ignore[attr-defined]
    assert len(head) <= 401  # 400 chars + "…"


def test_log_flow_respects_level():
    handler = _CapturingHandler()
    lg = logging.getLogger("flow.test.level")
    lg.addHandler(handler)
    lg.setLevel(logging.WARNING)

    log_flow("some.debug.stage", logger=lg, level=logging.DEBUG)
    assert len(handler.records) == 0  # DEBUG < WARNING, should be suppressed


# ---------------------------------------------------------------------------
# metrics.py — smoke test (no actual Prometheus scrape needed)
# ---------------------------------------------------------------------------

def test_metrics_observe_does_not_raise():
    from core.observability.metrics import (
        observe_turn, observe_llm_round, observe_tool_call,
        inc_guardrail_blocked, observe_worker_job, observe_mcp_tool,
    )
    observe_turn("chat", "ok", 1200)
    observe_llm_round("chat", 800, 320)
    observe_llm_round("chat", 800, None)  # ttft_ms can be None
    observe_tool_call("rag", "ok", 400)
    inc_guardrail_blocked("prompt_injection")
    observe_worker_job("indexing", "ok", 5000)
    observe_mcp_tool("my-server", "ok", 150)


# ---------------------------------------------------------------------------
# async context isolation (contextvars are per-task)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_isolated_across_tasks():
    import asyncio
    bind_context(turn_id="parent-turn")

    collected: list[str] = []

    async def child():
        # child inherits parent's context at create_task time
        fields = get_context_fields()
        collected.append(fields.get("turn_id", ""))
        # modifying context in child should not affect parent
        bind_context(turn_id="child-override")

    task = asyncio.create_task(child())
    await task

    # parent's turn_id is unchanged
    assert get_context_fields().get("turn_id") == "parent-turn"
    # child saw the parent's turn_id
    assert collected[0] == "parent-turn"
