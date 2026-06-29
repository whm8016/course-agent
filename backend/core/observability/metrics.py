"""业务级 Prometheus 指标（Phase 2）。

现有的 prometheus-fastapi-instrumentator（main.py L205）已自动暴露 HTTP 层指标
（请求延迟、状态码、in-flight 数），本模块在其基础上追加业务语义指标。

用法：
    from core.observability.metrics import (
        observe_turn_duration, inc_guardrail_blocked,
        observe_llm_ttft, observe_tool_call,
    )

查看指标（无需 Grafana）：
    curl http://localhost:8002/metrics | grep ca_

说明：所有指标以 ca_ 前缀（course_agent）避免与 fastapi-instrumentator 冲突。
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

# --------------------------------------------------------------------------
# Turn（整体回合）
# --------------------------------------------------------------------------

TURN_DURATION = Histogram(
    "ca_turn_duration_seconds",
    "Total wall-clock time for a single agent turn",
    labelnames=["mode", "status"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)

# --------------------------------------------------------------------------
# Agent Loop（每轮 LLM 调用）
# --------------------------------------------------------------------------

LLM_ROUND_DURATION = Histogram(
    "ca_llm_round_duration_seconds",
    "Duration of a single LLM streaming round in the agent loop",
    labelnames=["mode"],
    buckets=(0.2, 0.5, 1, 2, 5, 15, 30),
)

LLM_FIRST_TOKEN = Histogram(
    "ca_llm_first_token_seconds",
    "Time-to-first-token (TTFT) for LLM streaming",
    labelnames=["mode"],
    buckets=(0.1, 0.2, 0.5, 1, 2, 5),
)

# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

TOOL_CALL_DURATION = Histogram(
    "ca_tool_call_duration_seconds",
    "Duration of individual tool invocations",
    labelnames=["tool_name", "status"],
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 15),
)

# --------------------------------------------------------------------------
# Safety guardrail
# --------------------------------------------------------------------------

GUARDRAIL_BLOCKED = Counter(
    "ca_guardrail_blocked_total",
    "Number of requests blocked by the safety guardrail",
    labelnames=["risk_type"],
)

# --------------------------------------------------------------------------
# Worker jobs
# --------------------------------------------------------------------------

WORKER_JOB_DURATION = Histogram(
    "ca_worker_job_duration_seconds",
    "Duration of background worker jobs (indexing, summaries)",
    labelnames=["job_type", "status"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600),
)

# --------------------------------------------------------------------------
# MCP tool invocations
# --------------------------------------------------------------------------

MCP_TOOL_DURATION = Histogram(
    "ca_mcp_tool_duration_seconds",
    "Duration of MCP tool invocations",
    labelnames=["server", "status"],
    buckets=(0.05, 0.2, 0.5, 1, 2, 5, 15),
)


# --------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------

def observe_turn(mode: str, status: str, elapsed_ms: int) -> None:
    TURN_DURATION.labels(mode=mode, status=status).observe(elapsed_ms / 1000)


def observe_llm_round(mode: str, elapsed_ms: int, ttft_ms: int | None) -> None:
    LLM_ROUND_DURATION.labels(mode=mode).observe(elapsed_ms / 1000)
    if ttft_ms is not None:
        LLM_FIRST_TOKEN.labels(mode=mode).observe(ttft_ms / 1000)


def observe_tool_call(tool_name: str, status: str, elapsed_ms: int) -> None:
    TOOL_CALL_DURATION.labels(tool_name=tool_name, status=status).observe(elapsed_ms / 1000)


def inc_guardrail_blocked(risk_type: str) -> None:
    GUARDRAIL_BLOCKED.labels(risk_type=risk_type).inc()


def observe_worker_job(job_type: str, status: str, elapsed_ms: int) -> None:
    WORKER_JOB_DURATION.labels(job_type=job_type, status=status).observe(elapsed_ms / 1000)


def observe_mcp_tool(server: str, status: str, elapsed_ms: int) -> None:
    MCP_TOOL_DURATION.labels(server=server, status=status).observe(elapsed_ms / 1000)
