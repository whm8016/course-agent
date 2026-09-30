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

from prometheus_client import REGISTRY, Counter, Gauge, Histogram


def _metric(factory, name: str, *args, **kwargs):
    """模块级指标幂等构造：同名已注册则复用，避免单进程内重复执行模块体时抛 Duplicated timeseries。

    根因：本模块在 import 时构造全局指标，全部注册进 prometheus_client 默认 REGISTRY（全进程共享）。
    生产 gunicorn 每 worker 是 fork 出的独立进程、各自 REGISTRY，不触发；但长跑单进程（ARQ worker、
    离线评测 inspect 单进程）若因任意原因二次执行本模块体——部分导入中途失败被 sys.modules 驱逐后
    ARQ 重试（max_tries）/ 第二模块身份 / reload——第二次 register 同名指标即抛
    ``Duplicated timeseries``，连累索引任务全 ERROR（scripts/eval_capabilities/run.py 曾用 monkeypatch
    绕过同一根因）。本 helper 把"已注册即复用"下沉到指标定义点，根治该类问题且正路径行为零变化。
    """
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return factory(name, *args, **kwargs)


# --------------------------------------------------------------------------
# Turn（整体回合）
# --------------------------------------------------------------------------

TURN_DURATION = _metric(
    Histogram,
    "ca_turn_duration_seconds",
    "Total wall-clock time for a single agent turn",
    labelnames=["mode", "status"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)

# 失败 turn 计数（错误处理与可观测性重整）：turn_runtime 记账点在 finally 统一打点
# （单一记账点，code 取错误信封 code，缺省 internal）。cancelled 单独走 status 不进本表。
TURN_ERRORS = _metric(
    Counter,
    "ca_turn_errors_total",
    "Failed turns by mode and error code (envelope code, see observability/errors.py)",
    labelnames=["mode", "code"],
)

# --------------------------------------------------------------------------
# Agent Loop（每轮 LLM 调用）
# --------------------------------------------------------------------------

LLM_ROUND_DURATION = _metric(
    Histogram,
    "ca_llm_round_duration_seconds",
    "Duration of a single LLM streaming round in the agent loop",
    labelnames=["mode"],
    buckets=(0.2, 0.5, 1, 2, 5, 15, 30),
)

LLM_FIRST_TOKEN = _metric(
    Histogram,
    "ca_llm_first_token_seconds",
    "Time-to-first-token (TTFT) for LLM streaming",
    labelnames=["mode"],
    buckets=(0.1, 0.2, 0.5, 1, 2, 5),
)

# LLM token 用量（成本可观测性）。token_type ∈ input/output/cache_read。
# 逐轮在 _one_round 埋（core/observability/cost.observe_usage）。命名对齐 OTel GenAI 语义约定。
LLM_TOKENS_TOTAL = _metric(
    Counter,
    "ca_llm_tokens_total",
    "LLM token consumption by type (input/output/cache_read)",
    labelnames=["model", "token_type"],
)

# LLM 成本（按 run_agent_loop 调用汇总，单位美元）。cost = estimate_cost(model, 累积usage)。
# course_id/mode label 在 loop 层才有 context，故在 run_agent_loop 的 done 处汇总埋
# （core/observability/cost.observe_cost），不在逐轮的 _one_round 埋。
LLM_COST_USD_TOTAL = _metric(
    Counter,
    "ca_llm_cost_usd_total",
    "Estimated LLM cost in USD (aggregated per agent loop)",
    labelnames=["model", "course_id", "mode"],
)

# LLM 真实 usage 缺失计数（usage_missing）。端点不支持 stream_options 被剥参重试时该轮
# token 记 0（缺失≠真实 0，缺失不补估算只可观测——见 cost.observe_usage_missing）。缺失率
# 突增 = 端点行为变化信号。reason=stream_options_unsupported（llm.py / agent_base.py 剥参点）；
# 原因不明的缺失不打本 Counter（防与剥参点双计），经 TokenUsage.samples=0 可观测。
LLM_USAGE_MISSING_TOTAL = _metric(
    Counter,
    "ca_llm_usage_missing_total",
    "LLM rounds where real usage was not collected (recorded as 0, not estimated)",
    labelnames=["model", "reason"],
)

# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

TOOL_CALL_DURATION = _metric(
    Histogram,
    "ca_tool_call_duration_seconds",
    "Duration of individual tool invocations",
    labelnames=["tool_name", "status"],
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 15),
)

# --------------------------------------------------------------------------
# Worker jobs
# --------------------------------------------------------------------------

WORKER_JOB_DURATION = _metric(
    Histogram,
    "ca_worker_job_duration_seconds",
    "Duration of background worker jobs (indexing, summaries)",
    labelnames=["job_type", "status"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600),
)

# --------------------------------------------------------------------------
# MCP tool invocations
# --------------------------------------------------------------------------

MCP_TOOL_DURATION = _metric(
    Histogram,
    "ca_mcp_tool_duration_seconds",
    "Duration of MCP tool invocations",
    labelnames=["server", "status"],
    buckets=(0.05, 0.2, 0.5, 1, 2, 5, 15),
)


# --------------------------------------------------------------------------
# Leader election（多 worker 单例收敛：Cron/Bot/MCP 仅 leader 运行）
# --------------------------------------------------------------------------

LEADER_STATUS = _metric(
    Gauge,
    "ca_leader_is_leader",
    "1 if this worker currently holds the leader lock (runs Cron/Bot/MCP)",
    labelnames=["worker_id"],
)


# --------------------------------------------------------------------------
# 资源水位（压测/运维观测：DB 连接池 + LightRAG 实例池饱和度）
# 由 main.py 的 _sample_resource_gauges 后台 task 每 5s 写入。worker label 用 pid
# 区分（每进程独立 pool + 实例池；多容器/多 worker 天然隔离）。
# --------------------------------------------------------------------------

DB_POOL_CHECKEDOUT = _metric(
    Gauge,
    "ca_db_pool_checkedout",
    "SQLAlchemy engine pool: checked-out (in-use) connections",
    labelnames=["worker"],
)
# 这两个 Gauge 无 label：多 worker（gunicorn）下所有进程写同一 labelset，默认 'all' 会碰撞。
# multiprocess_mode='livesum' 让 MultiProcessCollector 跨存活 worker 求和（总量），且自动排除
# 已退出 worker（'live' 前缀）。pid-labeled 的 LEADER_STATUS / DB_POOL 因 labelset 各异，用默认 'all'。
LIGHTRAG_INSTANCES = _metric(
    Gauge,
    "ca_lightrag_instances",
    "LightRAG instance pool: cached instances count",
    multiprocess_mode="livesum",
)
LIGHTRAG_IN_USE = _metric(
    Gauge,
    "ca_lightrag_in_use",
    "LightRAG instance pool: in-use (referenced) instances count",
    multiprocess_mode="livesum",
)


# --------------------------------------------------------------------------
# LLM 可靠性（熔断 / 重试 / fallback）。reliability.py 在状态转换处写 gauge/counter，
# llm.py 在 fallback 分支写 fallback 计数。多 worker 各进程独立熔断器 → state 用
# liveall（每存活 worker 一条序列，Prometheus 取 max 即"是否有进程处于 OPEN"）。
# --------------------------------------------------------------------------
# 熔断器状态：0=closed 1=half_open 2=open（label breaker = 注册名，如 binding 名
# dashscope / asr:paraformer / vision:qwen-vl-plus / topic_build）。
CIRCUIT_STATE = _metric(
    Gauge,
    "ca_llm_circuit_state",
    "LLM circuit breaker state (0=closed 1=half_open 2=open)",
    labelnames=["breaker"],
    multiprocess_mode="liveall",
)
CIRCUIT_TRANSITIONS = _metric(
    Counter,
    "ca_llm_circuit_transitions_total",
    "LLM circuit breaker state transitions",
    labelnames=["breaker", "to"],
)
# 重试计数：with_retry_and_circuit 每调度一次重试 inc 1（不含首发）。
LLM_RETRIES = _metric(
    Counter,
    "ca_llm_retries_total",
    "LLM call retries scheduled by with_retry_and_circuit",
)
# fallback 触发计数：主路径熔断/重试耗尽后切 fallback 端点，outcome=success/failure。
LLM_FALLBACK = _metric(
    Counter,
    "ca_llm_fallback_triggers_total",
    "LLM fallback endpoint triggers (outcome: success/failure)",
    labelnames=["outcome"],
)


# --------------------------------------------------------------------------
# 磁盘水位 + 存储 GC（运维观测：派生数据体积、整卷水位、GC 回收量）
# DISK_USED_BYTES：整卷水位（main.py _sample_resource_gauges 每 5s 写一次 shutil.disk_usage）。
#   multiprocess_mode='liveall'：多 worker 各采一次，Prometheus 取 max（同一卷，值应一致）。
# STORAGE_GC_DELETED_BYTES：GC 累计回收字节数（run_gc 每次跑完 inc 总回收量）。
# --------------------------------------------------------------------------

DISK_USED_BYTES = _metric(
    Gauge,
    "ca_disk_used_bytes",
    "Disk volume usage in bytes (whole mounted volume)",
    labelnames=["target"],
    multiprocess_mode="liveall",
)
STORAGE_GC_DELETED_BYTES = _metric(
    Counter,
    "ca_storage_gc_deleted_bytes_total",
    "Total bytes reclaimed by storage GC (offline cron + admin trigger)",
)


# --------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------

def observe_turn(mode: str, status: str, elapsed_ms: int) -> None:
    TURN_DURATION.labels(mode=mode, status=status).observe(elapsed_ms / 1000)


def observe_turn_error(mode: str, code: str) -> None:
    """失败 turn 计数（status=error 时由 turn_runtime finally 与 observe_turn 并排调）。"""
    TURN_ERRORS.labels(mode=mode, code=code).inc()


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
