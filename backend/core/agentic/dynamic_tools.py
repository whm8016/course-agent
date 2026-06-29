"""DynamicToolResolver — 取代 ``_get_tool_schemas`` 调用的动态 schema 组装层。

组装本轮完整 tool schema list：``base``（静态 schema 按 enabled_tools 过滤）
+ ``read_skill``（skills_manifest 非空时）+ ``load_tools``（有 deferred pool 时）
+ 本会话已 load 的 ``mcp_*`` schema。返回的 list 是**可变**的，并经
``DeferredToolLoader.bind_live_schemas`` 绑定——``run_agent_loop`` 每轮复用同一 list
引用（``loop.py:252`` ``schemas = None if final else tool_schemas``），故 ``load()``
的 ``.append()`` 在下一轮立即可见，**核心循环零改动**。

loader 经 contextvar 暴露给 ``execute_tool("load_tools")``（和 solve session_id 同模式）。
"""
from __future__ import annotations

import contextvars
from typing import Any

from core.context import UnifiedContext
from core.observability import log_flow

# load_tools 是动态 schema（deferred pool 非空时挂载）。对标 DeepTutor LoadToolsTool。
LOAD_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_tools",
        "description": (
            "加载扩展工具（如 MCP 服务器工具），使其在本会话中可调用。"
            "当任务需要「扩展工具」清单中尚未加载的工具时使用；未加载就直接调用扩展工具会失败。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "description": "工具名数组，须与「扩展工具」清单完全一致",
                    "items": {"type": "string"},
                },
            },
            "required": ["names"],
        },
    },
}

# 当前 turn 的 DeferredToolLoader（execute_tool("load_tools") 经此取）
_CURRENT_LOADER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "deferred_loader", default=None
)


def current_deferred_loader() -> Any:
    return _CURRENT_LOADER.get()


def reset_deferred_loader(token: Any) -> None:
    """turn 结束时清理 loader（token 来自 resolve 返回值）。"""
    if token is not None:
        _CURRENT_LOADER.reset(token)


def resolve(
    context: UnifiedContext,
) -> tuple[list[dict[str, Any]] | None, Any]:
    """组装本轮 tool_schemas（可变 list）+ 绑 deferred loader + set contextvar。

    返回 ``(schemas_or_None, reset_token)``：调用方在 turn 结束时对非 None token 调
    ``_CURRENT_LOADER.reset(token)``（contextvar task-local，正常情况下同一 task 内
    reset 等价于清理；保留以兼容嵌套）。
    """
    # base：内置 schema 按 enabled_tools 过滤（registry 为内置 schema 单一数据源）
    from core.agent.registry import get_tool_registry
    _registry = get_tool_registry()
    if context.enabled_tools:
        base: list[dict[str, Any]] = _registry.schemas_for(context.enabled_tools)
    else:
        base = []

    # read_skill（skills_manifest 非空时）
    if context.skills_manifest:
        base.append(_registry.get("read_skill").schema)
        # write_memory 始终挂载（用户明确说偏好时用）
    if context.user_id:
        wm = _registry.get("write_memory")
        if wm:
            base.append(wm.schema)

    # read_memory 按需挂载（有记忆内容时才挂）
    if context.user_id and context.metadata.get("has_memory"):
        rm = _registry.get("read_memory")
        if rm:
            base.append(rm.schema)

    # deferred MCP 工具（渐进式揭示）
    token = None
    try:
        from core.mcp.deferred_tools import (
            DeferredToolLoader,
            render_deferred_tools_manifest,
        )
        from core.mcp.manager import get_mcp_manager
        from core.mcp.session_state import load_loaded_tools

        pool = get_mcp_manager().tool_adapters_for_user(
            context.metadata.get("mcp_enabled_servers")
        )
        if pool:
            loader = DeferredToolLoader(
                pool=pool,
                session_id=context.session_id,
                loaded=load_loaded_tools(context.session_id),
            )
            base.extend(loader.initial_schemas())
            if loader.has_loadable():
                base.append(LOAD_TOOLS_SCHEMA)
            context.extended_tools_manifest = render_deferred_tools_manifest(pool)
            schemas = list(base)
            loader.bind_live_schemas(schemas)
            token = _CURRENT_LOADER.set(loader)
            log_flow("tools.resolve", enabled_tools=context.enabled_tools,
                     base_count=len(base), has_deferred=True, mcp_pool=len(pool))
            return schemas or None, token
    except Exception:
        # MCP 未就绪不阻塞 chat
        pass

    log_flow("tools.resolve", enabled_tools=context.enabled_tools,
             base_count=len(base), has_deferred=False)
    return list(base) or None, token


__all__ = [
    "LOAD_TOOLS_SCHEMA",
    "current_deferred_loader",
    "reset_deferred_loader",
    "resolve",
]
