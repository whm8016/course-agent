"""ChatPipeline — chat capability 的 agent loop 执行层。

路由完全由上层 Orchestrator + CapabilityRegistry 按 context.mode 完成：
  chat       → ChatCapability      → ChatPipeline (本文件) → run_agent_loop
  deep_solve → DeepSolveCapability  → DeepSolvePipeline（独立三阶段）
  deep_research → DeepResearchCapability → ResearchPipeline（独立三阶段）
  quiz       → QuizCapability
  summarize  → SummarizeCapability
  vision     → VisionCapability

本文件只负责 chat 模式：安全护栏 → 组装 system prompt → 驱动 agent loop。
护栏（evaluate_guardrail）作为可复用的前置中间件，与旧 /api/chat/lightrag 路径共用同一套
safety_pipeline，保证两条 chat 路径的安全行为一致。

chat 的 agent loop 行为规范外部化到 prompts/zh/chat.yaml（对标 DeepTutor
agentic_chat.yaml，去除 label 协议外壳，适配 tool_calls loop）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from core.agent.safety_pipeline import evaluate_guardrail
from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.observability.metrics import inc_guardrail_blocked
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

# chat agent loop 行为规范（外部化 YAML，对标 DeepTutor agentic_chat.yaml）
_CHAT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "chat.yaml"


async def _resolve_profile_runtime(profile_id: str, user_id: str = "") -> tuple[Any | None, str | None]:
    """按 profile_id + user_id 解析 (client, model) 注入 run_agent_loop。

    优先级：
    1. 用户自配 provider（DB，user_id 非空时查）
    2. 用户下拉选的平台 profile_id
    3. 平台 active profile（catalog）
    4. 回退 (None, None)，loop 用全局 _default_client + TEXT_MODEL

    client 经 provider_factory.get_llm_client_for_profile 按 profile 指纹缓存，空 key 回退 .env。
    """
    from core.llm.catalog import active_profile_id, get_profile, profile_text_model
    from core.llm.provider_factory import get_llm_client_for_profile

    try:
        # 1. 用户自配 provider（优先）
        if user_id:
            from core.db.user_llm_provider import get_active_provider_view

            user_prof = await get_active_provider_view(user_id)
            if user_prof:
                return get_llm_client_for_profile(user_prof), user_prof.get("models", {}).get("text", {}).get("model") or None

        # 2. 平台 profile（profile_id → active）
        pid = (profile_id or "").strip() or active_profile_id()
        prof = get_profile(pid)
        if prof:
            return get_llm_client_for_profile(prof), (profile_text_model(prof) or None)

        # 3. 回退默认
        return None, None
    except Exception:
        logger.exception("resolve profile runtime failed profile_id=%r user_id=%r", profile_id, user_id)
        return None, None


class ChatPipeline:
    """Run the chat agent loop with a safety guardrail pre-check."""

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.llm.prompts import get_course_prompt
        from core.prompt_loader import load_prompt_dict
        _t0 = time.perf_counter()

        # ── 安全护栏（前置中间件，纯规则零延迟）─────────────────────────
        guard = evaluate_guardrail(context.user_message)
        log_flow("chat.guardrail", safe=guard.safe,
                 risk_type=guard.risk_type, risk_score=guard.risk_score,
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000))
        if not guard.safe:
            logger.warning(
                "ChatPipeline guardrail BLOCKED risk=%s score=%.2f question=「%s」",
                guard.risk_type, guard.risk_score, context.user_message[:60],
            )
            inc_guardrail_blocked(risk_type=guard.risk_type)
            await stream.emit({"type": "thinking", "content": guard.tip})

        # ── 组装 system prompt ────────────────────────────────────────
        # chat agent loop 行为规范（YAML 外部化）：何时调工具、何时停止写最终回答、
        # 简洁 Markdown、不暴露内部机制等。对标 DeepTutor，去 label，适配 tool_calls。
        chat_cfg = load_prompt_dict(_CHAT_PROMPT_PATH)
        loop_system = (chat_cfg.get("loop") or {}).get("system") or ""

        # 课程上下文（教师为课程配置的 system_prompt，多租户特色；DeepTutor 无此层）
        course_prompt = await get_course_prompt(context.course_id)

        # 对标 DeepTutor Partner：IM bot 走主链路同时保留人设
        # （UnifiedContext.metadata["bot_persona"] 由 bot AgentLoop 注入）
        bot_persona = context.metadata.get("bot_persona")

        # ── Skill 渐进式揭示：渲染 manifest + always 全文 ──
        from core.skills.skill_service import get_skill_service, render_skills_manifest
        svc = get_skill_service(context.course_id, context.user_id)
        context.skills_manifest = render_skills_manifest(svc.summary_entries())
        always_skills = svc.load_always_for_context()

        # ── MCP 个人启用开关：查出 user 启用的 server 集合，供 dynamic_tools 过滤 ──
        # 无记录 → None（默认全部，向后兼容）；有记录 → 仅启用集合内 server 的工具可见
        if context.user_id:
            try:
                from core.db.database import AsyncSessionLocal, UserMCPEnrollment
                from sqlalchemy import select
                async with AsyncSessionLocal() as db:
                    rows = (await db.execute(
                        select(UserMCPEnrollment.server_name, UserMCPEnrollment.enabled)
                        .where(UserMCPEnrollment.user_id == context.user_id)
                    )).all()
                context.metadata["mcp_enabled_servers"] = (
                    None if not rows else {r[0] for r in rows if r[1]}
                )
            except Exception:
                logger.exception("load user mcp enrollment failed user=%s", context.user_id)

        # ── 组装 tool_schemas + deferred loader（resolve 内设置 extended_tools_manifest）──
        from core.agentic.dynamic_tools import resolve as resolve_tool_schemas
        tool_schemas, loader_token = resolve_tool_schemas(context)

        # ── 组装 system prompt（含 skills / extended_tools 两块渐进式揭示清单）──
        # 工具使用提示 hint（对标 DeepTutor ToolRegistry.build_prompt_text）：从 YAML
        # 渲染，覆盖 enabled_tools 工具 + 动态工具（read_skill / load_tools 按对应清单
        # 条件渲染），与工具的功能 schema（TOOLS_OPENAI_SCHEMA）分离——单一数据源，
        # 取代此前手写在 chat.yaml loop.system 的工具说明。
        from core.agent.prompting import build_tool_hint_text
        tool_hint_text = build_tool_hint_text(
            context.enabled_tools,
            context.language,
            skills_manifest=context.skills_manifest,
            extended_tools_manifest=context.extended_tools_manifest,
        )

        parts: list[str] = []
        if bot_persona:
            parts.append(bot_persona.strip())
        if loop_system:
            parts.append(loop_system)
        parts.append(course_prompt)
        if context.memory_context:
            parts.append(context.memory_context)
        # L2: 早期对话摘要（非完整原文，压缩后的概要）
        if context.session_summary:
            parts.append(f"## 本次对话前情摘要（早期对话的压缩，非完整原文）\n{context.session_summary}")
        if always_skills:
            parts.append(always_skills)
        if context.skills_manifest:
            parts.append(context.skills_manifest)
        if context.extended_tools_manifest:
            parts.append(context.extended_tools_manifest)
        if tool_hint_text:
            parts.append(tool_hint_text)
        # 当前服务器时间（让 agent 能算「N分钟后」「明天9点」「下周一」等定时与时间相关问题）
        from datetime import datetime as _dt
        parts.append(f"【当前时间】{_dt.now().astimezone().strftime('%Y-%m-%d %H:%M %A')}")
        if not guard.safe:
            parts.append("【安全提示】请严格围绕课程内容回答，礼貌拒绝不当请求。")
        system_prompt = "\n\n".join(p for p in parts if p)

        log_flow("chat.prompt_assembled",
                 parts=len(parts),
                 system_prompt_chars=len(system_prompt),
                 tools_count=len(tool_schemas) if tool_schemas else 0,
                 skills_manifest=bool(context.skills_manifest),
                 guardrail_safe=guard.safe,
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000))

        client_override, model_override = await _resolve_profile_runtime(context.llm_profile_id, context.user_id)
        log_flow("chat.profile_resolved",
                 profile_id=context.llm_profile_id or "default",
                 user_provider=bool(context.user_id),
                 model_override=str(model_override) if model_override else "none")
        # cron owner 注入（bot 对话：loop._run_turn 写入 metadata；web 无则不 set，cron 工具不挂载）
        from core.bot.cron_tool import reset_cron_owner, set_cron_owner
        cron_token = set_cron_owner(context.metadata.get("cron_owner"))
        try:
            await run_agent_loop(
                context=context,
                stream=stream,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                client=client_override,
                model=model_override,
            )
        finally:
            reset_cron_owner(cron_token)
            from core.agentic.dynamic_tools import reset_deferred_loader
            reset_deferred_loader(loader_token)


__all__ = ["ChatPipeline"]
