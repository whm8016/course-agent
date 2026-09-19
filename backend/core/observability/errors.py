"""统一错误信封（错误处理与可观测性重整，2026-09）。

设计对标：
- LiteLLM exceptions.py：错误分类注册表（类型 → (code, 用户文案) 单一映射）。
- OpenTelemetry trace/exceptions 语义约定：异常在边界统一记录（type/message/stacktrace）。
  本仓的落点：normalize_error 纯函数供四边界（orchestrator / TrackedCapability /
  agent loop / turn_runtime）共用；**counter 打点与 turn_errors 落库只发生在
  turn_runtime._run_turn 事件循环的 ERROR 分支（单一记账点，天然只记一次）**。
- Sentry capture_exception→event_id→500 页渲染引用号：错误信封携带学生可报障的
  ref（turn_id[:8]，HTTP 层 request_id[:8]）——复用现有 contextvars，不另造 id 体系。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from core.observability.context import get_context_fields

logger = logging.getLogger(__name__)

# llm_unavailable 的用户文案单一真相源在本表；reliability.LLM_SERVICE_UNAVAILABLE
# 反向引用此常量（llm_user_message / llm.py 继续工作，不双写字面量）。
LLM_UNAVAILABLE_MESSAGE = "AI 服务暂时不可用，请稍后重试。"


@dataclass(frozen=True)
class ErrorSpec:
    """一个错误码的完整规格：机器可读 code + 学生可读中文文案。"""
    code: str
    user_message: str


ERROR_REGISTRY: dict[str, ErrorSpec] = {
    spec.code: spec
    for spec in (
        ErrorSpec("llm_unavailable", LLM_UNAVAILABLE_MESSAGE),
        ErrorSpec("context_overflow", "对话内容过长，请精简后重试或开启新会话。"),
        ErrorSpec("schema_invalid", "返回内容格式异常，请重试。"),
        ErrorSpec("tool_failed", "工具调用失败，请重试。"),
        ErrorSpec("capability_failed", "该功能暂时无法完成，请稍后重试。"),
        ErrorSpec("cancelled", "已停止生成。"),
        ErrorSpec("unauthorized", "没有权限执行此操作，请重新登录后再试。"),
        ErrorSpec("internal", "服务出了点问题，请稍后重试；若持续出现请联系老师并提供错误编号。"),
    )
}

ERROR_CODES = frozenset(ERROR_REGISTRY)

# 上下文超长的 provider 文案变体——单一真相源在本模块（分类是其职责）；loop 的
# 反应式裁剪判定（_is_context_length_error）反向 import 本表，勿在 loop 再抄一份。
CTX_OVERFLOW_KEYWORDS = (
    "context length", "maximum context", "too long", "reduce the length",
    "too many tokens", "input length", "exceeds the", "messages too",
)


def classify(exc: BaseException) -> ErrorSpec:
    """异常 → ErrorSpec（isinstance 链 → 状态码 → 供应商侧判定 → 兜底 internal）。

    tool_failed / capability_failed / unauthorized 由边界显式使用（调用方比类型
    判定更知情），classify 不产出它们。
    """
    # reliability 的异常类型延迟导入：reliability 模块级反向引用本表文案，
    # 模块级互相 import 会成环；函数内导入在调用期解析，无环。
    from core.llm.reliability import CircuitOpenError, LLMRetryError, classify_failure, status_of

    if isinstance(exc, asyncio.CancelledError):
        return ERROR_REGISTRY["cancelled"]
    if isinstance(exc, (CircuitOpenError, LLMRetryError)):
        return ERROR_REGISTRY["llm_unavailable"]
    # 上下文超长：413 或 provider 文案变体（与 loop 反应式兜底同判定口径）
    if status_of(exc) == 413:
        return ERROR_REGISTRY["context_overflow"]
    msg = str(exc).lower()
    if any(kw in msg for kw in CTX_OVERFLOW_KEYWORDS):
        return ERROR_REGISTRY["context_overflow"]
    if isinstance(exc, json.JSONDecodeError):
        return ERROR_REGISTRY["schema_invalid"]
    # 供应商侧故障（超时/连接/429/5xx——复用熔断重写的分类判定）统一按 LLM 不可用
    if classify_failure(exc):
        return ERROR_REGISTRY["llm_unavailable"]
    return ERROR_REGISTRY["internal"]


def error_ref() -> str:
    """学生可报障的引用号：turn_id 优先、request_id 回退，取前 8 位。

    8 位足够运维 grep 命中（ContextFilter 已把完整 turn_id/request_id 注入每行
    日志）；排障起点语义，非唯一键（turn_errors 表存完整 turn_id）。
    """
    fields = get_context_fields()
    rid = fields.get("turn_id") or fields.get("request_id") or ""
    return rid[:8]


def normalize_error(
    exc: BaseException, *, message_override: str = ""
) -> dict[str, str]:
    """异常 → 错误信封五元组（纯函数：不日志不打点，由调用方决定）。"""
    spec = classify(exc)
    return {
        "code": spec.code,
        "message": message_override or spec.user_message,
        "ref": error_ref(),
        "exc_type": type(exc).__name__,
        "exc_head": str(exc)[:200],  # 排障摘要；完整堆栈留在边界的 logger.exception
    }


async def emit_error_event(
    bus,
    exc: BaseException,
    *,
    source: str,
    message_override: str = "",
) -> dict[str, str]:
    """四边界共用的错误事件发射：现场记堆栈日志 + bus.error(完整信封) + 返回信封。

    emit 失败吞掉（错误处理路径绝不再抛）；调用方（如 turn_runtime 的 generator
    except）可直接用返回值赋本地五元组。
    """
    envelope = normalize_error(exc, message_override=message_override)
    logger.error(
        "error_event code=%s ref=%s source=%s exc_type=%s: %s",
        envelope["code"], envelope["ref"], source, envelope["exc_type"], exc,
        exc_info=exc,  # 堆栈只在抛错现场有——OTel「边界记录」的落点
    )
    try:
        await bus.error(
            envelope["message"],
            source=source,
            code=envelope["code"],
            ref=envelope["ref"],
            exc_type=envelope["exc_type"],
            exc_head=envelope["exc_head"],
        )
    except Exception:  # pragma: no cover - 错误路径的 emit 失败不能再抛
        logger.warning("emit_error_event: bus.error 发射失败", exc_info=True)
    return envelope

