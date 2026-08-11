"""Agent 显式读写 L3 语义记忆（mem0）的内置工具：write_memory / read_memory。

设计要点（对照 core/academic/tools.py 的读写分离与 IDOR 防线）：
  - 身份只用 registry 注入的 ``user_id``（``core/agent/registry.py``），schema **绝不**
    暴露 user_id 参数--若模型幻觉出该参数，``tool_dispatch.py`` 的 ``**call_kwargs``
    会与注入的 user_id 撞成 TypeError，被 registry 兜底成「工具执行失败」（安全但报错
    不友好，故 schema 干脆不出现身份参数），与三个学业工具同款 fail-closed 守卫。
  - write_memory 用 ``infer=False`` 存原文，不让 mem0 的 LLM 事实抽取改写/截断字面值
    （学生说「记一下我的会员卡号 6225...」必须逐字存）。mem0 ``add`` 内部先把 str
    归一成 ``[{"role":"user","content":...}]``，再走 ``_add_to_vector_store`` 的
    ``infer=False`` 分支逐字入库，不改写不截断。后台自动巩固路径（consolidation.py
    ``_promote_segment``）仍走 ``infer=True``（默认）做提炼，两条路径职责分明：
    自动路径提炼、显式路径存证。
  - read_memory 走 ``mem0.search``（空 query 退化 ``get_all`` 列最近若干条），filters
    强制带 ``user_id``（多租户隔离）+ ``course_id``（课程隔离，空则跨课程全局）。
  - 删除/修改不在工具范围--破坏性操作只走带 JWT 角色校验的 REST API
    ``/api/memory``（读写分离，OWASP LLM06），与学业工具「只读 SELECT」同一原则。

挂载：web 对话（``api/chat.py`` ``_ALWAYS_ON_TOOLS`` 无条件追加到 enabled_tools 末尾）；
schema + executor 由 ``register_builtins`` 装配（``core/agent/registry.py``）。
"""
from __future__ import annotations

import logging
from typing import Any

from core.agent.tool_protocol import ToolResult

logger = logging.getLogger(__name__)

# 写入长度上限：防止模型把整段对话/长文档塞进记忆库撑爆 context 与 pgvector。
_MAX_WRITE_CHARS = 1000
# 读出条数默认值与上限。
_READ_DEFAULT_LIMIT = 5
_READ_MAX_LIMIT = 20


async def execute_write_memory(
    *, course_id: str = "", user_id: str = "", **kwargs: Any
) -> ToolResult:
    """显式写入一条记忆（``infer=False`` 存原文，不抽取改写）。

    学生说「帮我记一下 / 记住 / 别忘了」，或主动透露学习偏好、教材版本、目标分数等
    跨会话有用信息时，模型调用本工具逐字存证。
    """
    if not user_id:
        return ToolResult(content="当前会话无法确认学生身份，无法保存记忆。", success=False)

    content = str(kwargs.get("content") or "").strip()
    if not content:
        return ToolResult(content="（记忆内容为空，未保存）", success=False)
    truncated = len(content) > _MAX_WRITE_CHARS
    content = content[:_MAX_WRITE_CHARS]

    from core.memory.mem0_client import get_memory

    m = get_memory()
    # infer=False：逐字存原文，不跑 mem0 事实抽取（详见模块 docstring 的两路径分工）。
    await m.add(
        content,
        user_id=user_id,
        metadata={"course_id": course_id, "source": "explicit_tool"},
        infer=False,
    )
    note = "（内容过长已截断）" if truncated else ""
    logger.info(
        "[mem0-tool] write_memory user_id=%s course_id=%s len=%d truncated=%s",
        user_id, course_id, len(content), truncated,
    )
    return ToolResult(content=f"已记住{note}：{content}")


async def execute_read_memory(
    *, course_id: str = "", user_id: str = "", **kwargs: Any
) -> ToolResult:
    """语义检索学生本人的记忆（预注入「用户记忆」段没命中时的兜底再查）。

    query 非空走 ``mem0.search``；query 为空退化 ``get_all`` 列最近若干条。filters
    强制带 ``user_id``（多租户隔离）+ ``course_id``（课程隔离，空则跨课程全局）。
    """
    if not user_id:
        return ToolResult(content="当前会话无法确认学生身份，无法读取记忆。", success=False)

    query = str(kwargs.get("query") or "").strip()
    try:
        limit = int(kwargs.get("limit") or _READ_DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = _READ_DEFAULT_LIMIT
    limit = max(1, min(limit, _READ_MAX_LIMIT))

    from core.memory.mem0_client import get_memory, normalize_results

    m = get_memory()
    filters: dict[str, Any] = {"user_id": user_id}
    if course_id:
        filters["course_id"] = course_id

    try:
        if query:
            raw = await m.search(query, filters=filters, top_k=limit)
        else:
            raw = await m.get_all(filters=filters, top_k=limit)
    except Exception as exc:
        logger.warning("[mem0-tool] read_memory failed user_id=%s error=%s", user_id, exc)
        return ToolResult(content=f"（读取记忆失败：{exc}）", success=False)

    items = normalize_results(raw)

    if not items:
        scope = f"「{query}」" if query else ""
        return ToolResult(content=f"没有找到相关记忆{scope}。", success=False)

    lines = [f"找到 {len(items)} 条记忆："]
    for it in items:
        text = (it.get("memory") or "").strip() if isinstance(it, dict) else str(it)
        if text:
            lines.append(f"- {text}")
    logger.info(
        "[mem0-tool] read_memory user_id=%s query=%s count=%d",
        user_id, query[:50], len(items),
    )
    return ToolResult(content="\n".join(lines))


# ── OpenAI function schemas（不含任何身份参数）──────────────────────────────

WRITE_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_memory",
        "description": (
            "把一条信息写入学生的跨会话长期记忆。当学生明确要求「帮我记一下 / 记住 / 别忘了」，"
            "或主动透露学习偏好、教材版本、目标分数等下次还有用的信息时调用。会逐字保存原话，"
            "不做改写。只能写入当前学生本人的记忆。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要记住的内容，用学生原话或其明确要记的信息。最多 1000 字。",
                },
            },
            "required": ["content"],
        },
    },
}


READ_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": (
            "检索学生本人此前保存的长期记忆。当上下文里的「用户记忆」段没有所需信息、"
            "但学生的问法暗示之前提过时，换个说法再查一次。只能读当前学生本人的记忆。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或自然语言描述。不传则返回最近若干条记忆。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回的条目数，默认 5。",
                },
            },
            "required": [],
        },
    },
}


__all__ = [
    "execute_write_memory",
    "execute_read_memory",
    "WRITE_MEMORY_SCHEMA",
    "READ_MEMORY_SCHEMA",
]
