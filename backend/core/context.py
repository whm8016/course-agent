"""
Unified Context
===============

A single data object that flows from the API layer through the orchestrator
into every tool / pipeline invocation for a single user turn.

【架构角色】统一上下文载体：API 层构造，传入 run_agent_loop() / ChatCapability.run()
再被各 node 函数读取。一次用户回合的所有「输入面」信息集中在此 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.attachment import Attachment


@dataclass
class UnifiedContext:
    """Everything a pipeline node or tool needs to process a single user turn.

    Attributes:
        course_id: 课程标识符，用于 RAG 检索和系统提示词。
        user_id: 数据库中的用户 ID，用于记忆读写、图谱更新。
        session_id: 会话持久化标识（可选），供日志追踪使用。
        user_message: 当前用户输入文本。
        conversation_history: 历史消息列表（OpenAI message 格式）。
        image_path: 本轮上传的单张图片路径（无图片时为 None；向后兼容旧字段，
            loop/chat_stream 内部会自动合并进 attachments）。
        attachments: 本轮附件列表（Attachment，支持多图；新代码优先用此字段）。
        mode: 已 normalize 的交互模式（"chat" | "deep_solve" | "quiz" | "research" |
            "vision" | "summarize"），由 API 层调用 normalize_mode() 后写入。
        enabled_tools: 用户本轮启用的工具名列表（如 ["rag", "web_search"]）。
            空列表表示未启用任何可选工具。
        memory_context: 记忆快照文本，由 build_memory_context(user) 生成，
            注入 system prompt 使用。
        language: 响应语言（默认 "zh"）。
        skills_manifest: 渐进式揭示的 Skills 清单块（一行/skill）；非空时
            自动挂载 read_skill 工具。由 ChatPipeline 经 SkillService 渲染填入。
        extended_tools_manifest: 渐进式揭示的扩展工具（deferred MCP）清单块；
            非空时自动挂载 load_tools 工具（阶段3）。
        llm_profile_id: 本轮用户在对话下拉选中的 LLM 供应商 profile id（对标 DeepTutor
            per-request provider 切换）。空字符串表示未选择，走默认 / active profile。
            ChatPipeline 据此动态构造 client+model 注入 run_agent_loop（即时生效，不重启）。
        metadata: 兜底字典，供各 pipeline 阶段写入临时扩展字段。
    """

    course_id: str = ""
    user_id: str = ""
    session_id: str = ""
    user_message: str = ""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    image_path: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    mode: str = "chat"
    enabled_tools: list[str] = field(default_factory=list)
    memory_context: str = ""  # L3: mem0 事实记忆
    session_summary: str = ""  # L2: 早期对话摘要
    language: str = "zh"
    skills_manifest: str = ""
    extended_tools_manifest: str = ""
    llm_profile_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
