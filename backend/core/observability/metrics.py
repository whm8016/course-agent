"""业务级 Prometheus 指标（Phase 2）。

现有的 prometheus-fastapi-instrumentator（main.py L205）已自动暴露 HTTP 层指标
（请求延迟、状态码、in-flight 数），本模块在其基础上追加业务语义指标。

用法：
    from core.observability.metrics import (
        observe_turn, observe_llm_round, observe_tool_call,
    )

查看指标（无需 Grafana）：
    curl http://localhost:8002/metrics | grep ca_

说明：所有指标以 ca_ 前缀（course_agent）避免与 fastapi-instrumentator 冲突。
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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

# LLM token 用量（成本可观测性）。token_type ∈ input/output/cache_read。
# 逐轮在 _one_round 埋（core/observability/cost.observe_usage）。命名对齐 OTel GenAI 语义约定。
LLM_TOKENS_TOTAL = Counter(
    "ca_llm_tokens_total",
    "LLM token consumption by type (input/output/cache_read)",
    labelnames=["model", "token_type"],
)

# LLM 成本（按 run_agent_loop 调用汇总，单位美元）。cost = estimate_cost(model, 累积usage)。
# course_id/mode label 在 loop 层才有 context，故在 run_agent_loop 的 done 处汇总埋
# （core/observability/cost.observe_cost），不在逐轮的 _one_round 埋。
LLM_COST_USD_TOTAL = Counter(
    "ca_llm_cost_usd_total",
    "Estimated LLM cost in USD (aggregated per agent loop)",
    labelnames=["model", "course_id", "mode"],
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
# Leader election（多 worker 单例收敛：Cron/Bot/MCP 仅 leader 运行）
# --------------------------------------------------------------------------

LEADER_STATUS = Gauge(
    "ca_leader_is_leader",
    "1 if this worker currently holds the leader lock (runs Cron/Bot/MCP)",
    labelnames=["worker_id"],
)


# --------------------------------------------------------------------------
# 资源水位（压测/运维观测：DB 连接池 + LightRAG 实例池饱和度）
# 由 main.py 的 _sample_resource_gauges 后台 task 每 5s 写入。worker label 用 pid
# 区分（每进程独立 pool + 实例池；多容器/多 worker 天然隔离）。
# --------------------------------------------------------------------------

DB_POOL_CHECKEDOUT = Gauge(
    "ca_db_pool_checkedout",
    "SQLAlchemy engine pool: checked-out (in-use) connections",
    labelnames=["worker"],
)
# 这两个 Gauge 无 label：多 worker（gunicorn）下所有进程写同一 labelset，默认 'all' 会碰撞。
# multiprocess_mode='livesum' 让 MultiProcessCollector 跨存活 worker 求和（总量），且自动排除
# 已退出 worker（'live' 前缀）。pid-labeled 的 LEADER_STATUS / DB_POOL 因 labelset 各异，用默认 'all'。
LIGHTRAG_INSTANCES = Gauge(
    "ca_lightrag_instances",
    "LightRAG instance pool: cached instances count",
    multiprocess_mode="livesum",
)
LIGHTRAG_IN_USE = Gauge(
    "ca_lightrag_in_use",
    "LightRAG instance pool: in-use (referenced) instances count",
    multiprocess_mode="livesum",
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


def observe_worker_job(job_type: str, status: str, elapsed_ms: int) -> None:
    WORKER_JOB_DURATION.labels(job_type=job_type, status=status).observe(elapsed_ms / 1000)


def observe_mcp_tool(server: str, status: str, elapsed_ms: int) -> None:
    MCP_TOOL_DURATION.labels(server=server, status=status).observe(elapsed_ms / 1000)


def set_leader_status(worker_id: str, is_leader: bool) -> None:
    LEADER_STATUS.labels(worker_id=worker_id).set(1 if is_leader else 0)
