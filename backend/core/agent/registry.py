"""轻量 ToolRegistry — 函数式/数据式工具注册中心。

对标 DeepTutor ``ToolRegistry`` 的**架构思想**（单一注册表统一执行路由），
但不引入 BaseTool/ToolDefinition OOP 体系：executor 是 async callable，schema
是 dict，注册 ``ToolEntry`` 后 ``execute`` 查表调用。

职责边界（重要）：
- 内置工具：schema 与执行都由 registry 统一。
- MCP 工具：**仅执行路由**由 registry 接管（``execute`` 查表）；MCP schema 的
  渐进式揭示（``load_tools`` + turn-bound live list + session 持久化 + stale 清理）
  仍由 ``DeferredToolLoader`` 负责——registry 是全局静态表，无 turn-bound 语义，
  替代不了 loader 的揭示职责。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.agent.tool_protocol import ToolResult

logger = logging.getLogger(__name__)

# executor 统一签名：与 execute_tool 同形（course_id/user_id 为关键字，业务参数走 kwargs）
ToolExecutor = Callable[..., Awaitable[ToolResult]]


@dataclass
class ToolEntry:
    name: str
    schema: dict[str, Any]  # OpenAI function schema dict
    executor: ToolExecutor  # async (*, course_id, user_id, **kwargs) -> ToolResult
    deferred: bool = False  # MCP 工具默认 True（标记用，不影响 execute 路由）


class ToolRegistry:
    """查表式工具注册中心：register / get / schemas_for / execute。"""

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    # ── 注册 ────────────────────────────────────────────────────────────
    def register(self, entry: ToolEntry) -> None:
        # 幂等：同名覆盖（reload / 重复 register_builtins 安全）
        self._entries[entry.name] = entry

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def clear(self) -> None:
        self._entries.clear()

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    def has(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def schemas_for(self, names: list[str] | set[str] | None) -> list[dict[str, Any]]:
        """按名称集合过滤，返回命中的 schema（去重保序）；None/空 → []。"""
        if not names:
            return []
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            entry = self._entries.get(name)
            if entry is not None:
                seen.add(name)
                result.append(entry.schema)
        return result

    def deferred_entries(self) -> list[ToolEntry]:
        return [e for e in self._entries.values() if e.deferred]

    # ── 执行（核心）─────────────────────────────────────────────────────
    async def execute(
        self,
        name: str,
        *,
        course_id: str = "",
        user_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        entry = self._entries.get(name)
        if entry is None:
            return ToolResult(content=f"（未知工具：{name}）", success=False)
        try:
            # 关键字调用：course_id/user_id 命中具名参数或落进 **kwargs，业务参数透传
            return await entry.executor(course_id=course_id, user_id=user_id, **kwargs)
        except Exception as exc:
            logger.exception("registry execute failed: %s", name)
            return ToolResult(content=f"（工具执行失败：{exc}）", success=False)


def register_builtins(registry: ToolRegistry) -> None:
    """注册 8 个内置工具（6 静态 + read_skill + load_tools）成 ToolEntry。

    schema 取自 TOOLS_OPENAI_SCHEMA / READ_SKILL_SCHEMA / LOAD_TOOLS_SCHEMA，
    executor 取对应 _execute_*。deferred=False（内置工具非渐进式揭示；
    read_skill/load_tools 的"按需挂载"由 resolve 按 skills_manifest/manifest 决定，
    不靠 deferred 标记）。
    """
    from core.agent.tool_registry import (
        READ_SKILL_SCHEMA,
        TOOLS_OPENAI_SCHEMA,
        _execute_ask_user,
        _execute_load_tools,
        _execute_rag,
        _execute_read_skill,
        _execute_solve_finish_step,
        _execute_solve_plan,
        _execute_solve_replan,
        _execute_web_search,
    )
    from core.agentic.dynamic_tools import LOAD_TOOLS_SCHEMA
    from core.bot.cron_tool import CRON_SCHEMA, execute_cron

    executor_by_name: dict[str, ToolExecutor] = {
        "rag": _execute_rag,
        "web_search": _execute_web_search,
        "ask_user": _execute_ask_user,
        "read_skill": _execute_read_skill,
        "load_tools": _execute_load_tools,
        "solve_plan": _execute_solve_plan,
        "solve_finish_step": _execute_solve_finish_step,
        "solve_replan": _execute_solve_replan,
        "cron": execute_cron,
    }
    schema_by_name: dict[str, dict[str, Any]] = {
        s["function"]["name"]: s for s in TOOLS_OPENAI_SCHEMA
    }
    schema_by_name["read_skill"] = READ_SKILL_SCHEMA
    schema_by_name["load_tools"] = LOAD_TOOLS_SCHEMA
    schema_by_name["cron"] = CRON_SCHEMA

    for name, executor in executor_by_name.items():
        schema = schema_by_name.get(name)
        if schema is None:
            logger.warning("register_builtins: missing schema for %s", name)
            continue
        registry.register(ToolEntry(name=name, schema=schema, executor=executor, deferred=False))


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """返回进程级 ToolRegistry 单例（首次时注册内置工具）。"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        register_builtins(_registry)
    return _registry


__all__ = [
    "ToolEntry",
    "ToolExecutor",
    "ToolRegistry",
    "register_builtins",
    "get_tool_registry",
]
