"""Deferred tool loading（渐进式工具发现，对标 DeepTutor ``runtime/registry/deferred_tools.py``）。

MCP 工具（``deferred=True``）不进初始 tool 列表。system prompt 带一行/工具的 manifest
（:func:`render_deferred_tools_manifest`）；模型需要时调 ``load_tools``，
:class:`DeferredToolLoader` 把完整 schema append 到 live ``tool_schemas`` list——
``run_agent_loop`` 每轮重读该 list（``loop.py:252`` ``schemas = None if final else
tool_schemas``，同引用），工具立即可用。已加载名按 session 持久化，后续 turn 从一开始
就含这些 schema。

与 DeepTutor 的差异：DeepTutor 的 loader 经 ``ToolRegistry.get(name)`` 查 adapter；
本项目 registry 只接管**执行路由**（schema 揭示的 turn-bound 可变 list + session
持久化 + stale 清理语义 registry 替代不了），故 loader 仍持有 ``pool``
（wrapped_name → MCPToolAdapter 字典）直接查 schema。
"""
from __future__ import annotations

import logging
from typing import Any

from core.mcp.adapter import MCPToolAdapter

logger = logging.getLogger(__name__)


def render_deferred_tools_manifest(tools: list[MCPToolAdapter], *, language: str = "zh") -> str:
    """system-prompt 块：列出 deferred 工具，按 MCP server 分组。"""
    if not tools:
        return ""
    groups: dict[str, list[tuple[str, str]]] = {}
    for tool in tools:
        group = tool.server_name or "other"
        groups.setdefault(group, []).append((tool.wrapped_name, tool.description))
    lines = [
        "## 扩展工具",
        "这些工具存在但尚未加载；直接调用会失败。要使用其中任意工具，请先用准确的"
        "工具名调用 `load_tools`，随后这些 schema 会在本会话中保持可用。",
        "",
    ]
    for group in sorted(groups):
        header = f"### MCP 服务器：{group}" if group != "other" else "### 其他"
        lines.append(header)
        for name, desc in sorted(groups[group]):
            lines.append(f"- **{name}** — {desc}")
        lines.append("")
    return "\n".join(lines).rstrip()


class DeferredToolLoader:
    """per-turn 句柄：把 deferred 工具 schema 加载进 live list。

    由 ``DynamicToolResolver`` 每 turn 创建一次，经 contextvar 注入 ``load_tools`` 调用
    （LLM 看不到此句柄，只提供 ``names``）。
    """

    def __init__(
        self,
        *,
        pool: dict[str, MCPToolAdapter] | list[MCPToolAdapter] | None,
        session_id: str,
        loaded: set[str] | None = None,
        allowed: set[str] | None = None,
    ) -> None:
        if isinstance(pool, dict):
            self._pool: dict[str, MCPToolAdapter] = dict(pool)
        else:
            self._pool = {a.wrapped_name: a for a in (pool or [])}
        self._session_id = session_id
        self._loaded: set[str] = set(loaded or [])
        # None = 全部可加载（manager 已按 enabled_tools 过滤）；set = 收窄
        self._allowed = set(allowed) if allowed is not None else None
        self._live_schemas: list[dict[str, Any]] | None = None

    def _is_allowed(self, name: str) -> bool:
        return self._allowed is None or name in self._allowed

    @property
    def loaded_names(self) -> set[str]:
        return set(self._loaded)

    def has_loadable(self) -> bool:
        """是否有可加载的 deferred 工具（决定是否挂载 load_tools schema）。"""
        return bool(self._pool)

    def bind_live_schemas(self, schemas: list[dict[str, Any]]) -> None:
        """绑定本 turn 的 live ``tool_schemas`` list（原地 mutate）。"""
        self._live_schemas = schemas

    def initial_schemas(self) -> list[dict[str, Any]]:
        """本会话已加载工具的 schema（清理掉已失效的）。"""
        schemas: list[dict[str, Any]] = []
        stale: set[str] = set()
        for name in sorted(self._loaded):
            tool = self._pool.get(name)
            if tool is None or not getattr(tool, "deferred", False):
                stale.add(name)  # server 移除/改名 → 静默丢弃
                continue
            if not self._is_allowed(name):
                continue
            schemas.append(tool.to_openai_schema())
        if stale:
            self._loaded -= stale
            self._persist()
        return schemas

    def load(self, names: list[str]) -> dict[str, list[str]]:
        """加载给定 deferred 工具；按结果返回 name 列表。"""
        loaded: list[str] = []
        already: list[str] = []
        unknown: list[str] = []
        for raw in names:
            name = str(raw or "").strip()
            if not name:
                continue
            if name in self._loaded:
                already.append(name)
                continue
            tool = self._pool.get(name)
            if tool is None or not getattr(tool, "deferred", False) or not self._is_allowed(name):
                unknown.append(name)
                continue
            if self._live_schemas is not None:
                self._live_schemas.append(tool.to_openai_schema())
            self._loaded.add(name)
            loaded.append(name)
        if loaded:
            self._persist()
        return {"loaded": loaded, "already_loaded": already, "unknown": unknown}

    def _persist(self) -> None:
        try:
            from core.mcp.session_state import record_loaded_tools

            record_loaded_tools(self._session_id, self._loaded)
        except Exception:
            logger.warning("failed to persist deferred-tool state", exc_info=True)


__all__ = ["DeferredToolLoader", "render_deferred_tools_manifest"]
