"""chat_mode 规范化（mode_normalize）。

chat 路径走 core/agentic/loop.py（tool_calls 驱动的 Agent Loop）。
deep_solve / deep_research / quiz 各自走独立 Capability pipeline
（core/solve、core/research、core/question）。

本模块仅保留 mode 规范化函数：
  normalize_mode — API 层 chat_mode 规范化（api/chat、api/sessions 用）

历史遗留的 router_node（LLM 意图分类）、quiz_node（结构化出题）、OFF_TOPIC_REPLY，
以及 summarize/vision 独立流式节点，已被 tool_calls 机制与统一 chat pipeline 取代并移除。

命名说明：真正的能力路由器是 core/orchestrator.py（CourseOrchestrator）；本模块原叫
agent/orchestrator.py 但只含 mode 规范化，与 "orchestrator" 语义无关、易误导，故更名为
mode_normalize.py。
"""
from __future__ import annotations


def normalize_mode(mode: str | None) -> str:
    """规范化前端传入的 chat_mode，兼容旧值。"""
    allowed = {"chat", "deep_solve", "deep_research", "quiz"}
    if not mode:
        return "chat"
    normalized = mode.strip().lower()
    # 兼容旧前端传 "research"
    if normalized == "research":
        normalized = "deep_research"
    return normalized if normalized in allowed else "chat"
