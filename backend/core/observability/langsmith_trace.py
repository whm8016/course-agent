"""LangSmith 全链路追踪集中模块。

设计要点
========
1. 所有埋点经 ``is_tracing_enabled()`` 门控;未启用时 ``safe_traceable`` 退化为
   identity 装饰器(直接返回原函数,不创建 run、不 import langsmith 内部对象),
   ``trace_context`` 退化为空 contextmanager,``wrap_openai_client`` 原样返回 client。
   任何 langsmith 内部异常被 try/except 吞掉,绝不影响业务路径。
2. 启用判定统一从 ``config`` 读(LANGSMITH_TRACING + LANGSMITH_API_KEY),避免各文件
   单独 os.getenv 导致判定不一致。
3. 顶层 trace 用 ``trace_context()``:在 turn 外层建立 root run,下游 ``@traceable``
   函数与 ``wrap_openai`` 产生的 LLM run 通过 langsmith 的 contextvars run tree
   自动成为子 run,无需手动传 trace id。
4. LLM 层靠 ``wrap_openai_client``(对主/fallback/profile client 统一应用);非 LLM 层
   (工具 / RAG / LightRAG 内部 LLM)靠 ``safe_traceable``。两者各司其职,不叠加,
   避免 LLM 调用点产生双重 run。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, TypeVar

from settings import get_settings
LANGSMITH_API_KEY = get_settings().langsmith_api_key.get_secret_value()
LANGSMITH_PROJECT = get_settings().langsmith_project
LANGSMITH_TRACING = get_settings().langsmith_tracing

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 启动期一次性判定:tracing=true 且有 api key。
_TRACING_ON: bool = bool(LANGSMITH_TRACING) and bool(LANGSMITH_API_KEY)

# 延迟导入:未启用时根本不 import langsmith,降低冷启动开销与潜在副作用。
_traceable: Callable[..., Any] | None = None
if _TRACING_ON:
    try:
        from langsmith import traceable as _traceable  # type: ignore[import-untyped]
    except Exception:
        logger.exception("LangSmith traceable import failed; tracing disabled")
        _traceable = None
        _TRACING_ON = False


def is_tracing_enabled() -> bool:
    """运行期判断是否启用追踪(供非装饰器路径门控)。"""
    return _TRACING_ON


def safe_traceable(
    *,
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """``@traceable`` 的安全包装。

    用法::

        @safe_traceable(name="tool.execute", run_type="tool")
        async def execute_tool(...): ...

    未启用时返回 identity 装饰器(原函数透传,不创建 run)。
    启用时委托 ``langsmith.traceable``,并把 metadata/tags 透传。

    本项目所有 @traceable 埋点(execute_tool / _execute_rag / _llm_model_func)
    都是 async def 返回值,非 generator,无 async generator 上下文丢失风险。
    """

    def _identity(fn: Callable[..., T]) -> Callable[..., T]:
        return fn

    if not _TRACING_ON or _traceable is None:
        return _identity

    kwargs: dict[str, Any] = {"run_type": run_type}
    if name is not None:
        kwargs["name"] = name
    if metadata is not None:
        kwargs["metadata"] = metadata
    if tags is not None:
        kwargs["tags"] = tags
    try:
        return _traceable(**kwargs)
    except Exception:
        logger.exception("safe_traceable: traceable() apply failed; using identity")
        return _identity


@asynccontextmanager
async def trace_context(
    *,
    name: str = "turn",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> AsyncIterator[None]:
    """顶层 turn trace 的 async contextmanager。

    用在 ``_run_turn`` 外层:进入时建立顶层 run,退出时 end。本 context 内所有
    ``@traceable`` 函数与 ``wrap_openai`` 产生的 LLM run 自动成为子 run
    (langsmith run tree 经 contextvars 传递)。

    未启用时为 no-op(asynccontextmanager 空壳),零开销。
    """
    if not _TRACING_ON:
        yield
        return
    try:
        from langsmith import trace as _trace  # type: ignore[import-untyped]

        async with _trace(
            name=name,
            run_type="chain",
            metadata=metadata or {},
            tags=tags or [],
            project_name=LANGSMITH_PROJECT or None,
        ):
            yield
    except Exception:
        # 任何 langsmith 异常都不应阻断 turn 执行;降级为无 trace 继续运行。
        logger.exception("trace_context failed; continuing without trace")
        yield


def wrap_openai_client(client: Any, *, chat_name: str) -> Any:
    """对 AsyncOpenAI / AsyncAzureOpenAI 实例应用 ``wrap_openai``。

    用于统一包装主 client / fallback client / profile client。
    非 OpenAI 兼容实例(如 AnthropicAdapter)、None、或未启用时原样返回。
    """
    if not _TRACING_ON or client is None:
        return client
    try:
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        if isinstance(client, (AsyncOpenAI, AsyncAzureOpenAI)):
            from langsmith import wrappers  # type: ignore[import-untyped]

            return wrappers.wrap_openai(client, chat_name=chat_name)
    except Exception:
        logger.debug("wrap_openai_client '%s' failed", chat_name, exc_info=True)
    return client


_client_singleton: Any = None


def _get_client() -> Any:
    """懒单例 langsmith Client。

    每次 Client() 都新建 requests.Session（全新连接池）、注册 atexit、起一条批量
    tracing 守护线程——record_* 调用点已有多个，逐次新建是三重浪费；懒建一次复用。
    （GIL 下并发竞态最多重复建一次，无害。）
    """
    global _client_singleton
    if _client_singleton is None:
        from langsmith import Client  # type: ignore[import-untyped]

        _client_singleton = Client()
    return _client_singleton


def _record(
    *,
    name: str,
    run_type: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """手动记录一个独立 run(非装饰器场景)。

    挂在当前 run tree 下(若处于 trace_context 内则成为子 run,否则为顶层 run)。
    未启用时直接返回。
    """
    if not _TRACING_ON:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree  # type: ignore[import-untyped]

        parent = get_current_run_tree()
        client = _get_client()
        run_id = client.create_run(
            name=name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata or {},
            parent_run_id=getattr(parent, "id", None) if parent else None,
            project_name=LANGSMITH_PROJECT or None,
        )
        client.update_run(
            run_id,
            outputs=outputs or {},
            error=error,
        )
    except Exception:
        logger.debug("record run '%s' failed", name, exc_info=True)


def ls_extra(name: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造传给 ``client.chat.completions.create(**kwargs)`` 的 langsmith 附加参数。

    wrap_openai 包过的 client 识别 ``langsmith_extra``（给单次调用打名字/metadata，
    用于把压缩等辅助调用从同名的 course_agent_chat 流量里区分出来）；原生 openai
    client 不认识该 kwarg（直接 TypeError），故未启用追踪时返回空 dict，调用点
    ``create(**create_kwargs, **ls_extra(...))`` 逐字节等价于原调用。

    前提：``_TRACING_ON`` 与"client 已 wrap 成功"一致（llm.py import 期统一 wrap；
    wrap 失败仅 debug 日志）。升级 langsmith 若改 ``wrap_openai`` 行为需复核此门控。
    """
    if not _TRACING_ON:
        return {}
    extra: dict[str, Any] = {"name": name}
    if metadata:
        extra["metadata"] = metadata
    return {"langsmith_extra": extra}


def record_tool_result(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    content: str,
    success: bool,
    elapsed_ms: int,
    sources: list[dict] | None = None,
) -> None:
    """记录一次工具调用结果(用于不便用装饰器的手动场景)。"""
    _record(
        name=f"tool.{tool_name}",
        run_type="tool",
        inputs={"arguments": arguments},
        outputs={
            "content_chars": len(content),
            "content_preview": content[:1000],
            "success": success,
            "elapsed_ms": elapsed_ms,
            "sources": sources or [],
        },
        metadata={"tool": tool_name},
        error=None if success else "tool_error",
    )


def record_rag_result(
    *,
    course_id: str,
    query: str,
    mode: str,
    retrieved_chars: int,
    preview: str,
    sources: list[dict] | None = None,
) -> None:
    """记录一次 RAG 检索结果。"""
    _record(
        name="rag.retrieve",
        run_type="retriever",
        inputs={"course_id": course_id, "query": query, "mode": mode},
        outputs={
            "retrieved_chars": retrieved_chars,
            "preview": preview[:1000],
            "sources": sources or [],
        },
        metadata={"course_id": course_id, "mode": mode},
    )


def record_context_trim(
    *,
    stage: str,
    coordinator_enabled: bool,
    iteration: int,
    dropped_tool_results: int = 0,
    masked_turns: int = 0,
    dropped_messages: int = 0,
    summary_added: bool = False,
    dropped_preview: str | None = None,
    summary_text: str | None = None,
) -> None:
    """记录一次上下文裁剪（轮内 enforce / 回合前 plan_turn 裁剪）。

    轮内裁剪没有天然对应的 LLM/tool run，此前完全不可观测——LangSmith 只能看到裁剪
    *之后* 发给模型的 messages，看不到这一步删了什么。挂在当前 turn run 下为子 run。

    dropped_preview/summary_text 补正文截断（此前只留计数/布尔，"压缩成了什么"在
    LangSmith 上无从查证）；统一在此截断，调用方无需各自把关。
    """
    if not _TRACING_ON:
        return  # 关态早退：调用方拼 preview/join 的开销在 _emit_trim_report 侧已省，这里再省 outputs 构建
    _record(
        name=f"context.trim.{stage}",
        run_type="chain",
        inputs={"coordinator_enabled": coordinator_enabled, "iteration": iteration},
        outputs={
            "dropped_tool_results": dropped_tool_results,
            "masked_turns": masked_turns,
            "dropped_messages": dropped_messages,
            "summary_added": summary_added,
            "dropped_preview": (dropped_preview or "")[:800],
            "summary_text": (summary_text or "")[:800],
        },
    )


__all__ = [
    "is_tracing_enabled",
    "safe_traceable",
    "trace_context",
    "wrap_openai_client",
    "record_tool_result",
    "record_rag_result",
    "record_context_trim",
    "ls_extra",
]
