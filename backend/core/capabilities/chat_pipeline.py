"""ChatPipeline — chat capability 的 agent loop 执行层。

路由完全由上层 Orchestrator + CapabilityRegistry 按 context.mode 完成：
  chat          → ChatCapability         → ChatPipeline (本文件) → run_agent_loop
  deep_solve    → DeepSolveCapability    → DeepSolvePipeline（独立三阶段）
  deep_research → DeepResearchCapability → ResearchPipeline（独立三阶段）
  quiz          → QuizCapability

本文件只负责 chat 模式：组装 system prompt → 驱动 agent loop。

chat 的 agent loop 行为规范外部化到 prompts/zh/chat.yaml（agentic_chat.yaml，去除 label 协议外壳，适配 tool_calls loop）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.pipeline_common import (
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
)
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

# chat agent loop 行为规范（外部化 YAML，agentic_chat.yaml）
_CHAT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "chat.yaml"


def assemble_system_prompt(
    *,
    loop_system: str,
    course_prompt: str,
    bot_persona: str = "",
    always_skills: str = "",
    skills_manifest: str = "",
    tool_hint_text: str = "",
    extended_tools_manifest: str = "",
    memory_context: str = "",
    session_summary: str = "",
    now_text: str = "",
) -> str:
    """按「稳定性递减」拼装 chat system prompt，最大化 prefix cache 命中。

    deepseek/qwen 的 prefix cache 要求多次请求的 system prompt 前缀逐字一致才命中
    （langsmith 里看 cached_tokens）。越稳定的段越靠前，同课程/同用户的多轮请求就能
    共享越长的前缀。各段稳定性依据（见 ChatPipeline.run 取值 + build_tool_hint_text）：

      L0 全局稳定 : loop_system      — chat.yaml，所有请求逐字一致（最稳，放最前）
      L1 课程级   : course_prompt    — 教师为课程配的 system_prompt，同课程一致
      L1 课程级   : bot_persona      — web 恒空(一致)；仅 IM bot 注入非空人设
      L1 课程级   : always_skills    — always-on skill 全文，课程配置为主
      L1/用户     : skills_manifest  — skill 清单，课程级为主；建了 personal skill 的用户分叉
      L2 用户级   : tool_hint_text   — 依赖 enabled_tools + skills/extended 的「有无」
                                       (build_tool_hint_text 仅按有无追加 read_skill/load_tools
                                       提示)，故放在 skills 之后、extended 之前——extended
                                       内容变化不波及更靠前 tool_hint 的 cache 命中
      L2 用户级   : extended_tools_manifest — MCP 个人启用集合，内容随用户变
      L2 用户级   : memory_context   — 每用户每轮可能不同
      L2 用户级   : session_summary  — 每隔几轮才更新
      L3 每请求   : now_text         — 当前时间，永远最后

    纯函数：只做「定序 + 过滤空段 + 以空行连接」，不读 DB/时间/配置/格式化，便于
    行为测试（tests/test_prompt_assembly.py）。
    """
    parts = [
        loop_system,
        course_prompt,
        bot_persona,
        always_skills,
        skills_manifest,
        tool_hint_text,
        extended_tools_manifest,
        memory_context,
        session_summary,
        now_text,
    ]
    return "\n\n".join(p for p in parts if p)


class ChatPipeline:
    """Run the chat agent loop."""

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.prompt_loader import load_prompt_dict
        _t0 = time.perf_counter()

        # ── 解析对话供应商 runtime（client / text_model / binding）─────────────
        # 四条 pipeline 共享步骤（pipeline_common）：用户自配 > 平台 profile > 全 None
        # （loop 回退全局 _default_client + TEXT_MODEL）。binding 决定图片直注格式。
        rt = await resolve_profile_runtime(context.llm_profile_id, context.user_id)
        log_flow("chat.profile_resolved",
                 profile_id=context.llm_profile_id or "default",
                 user_provider=bool(context.user_id),
                 model_override=str(rt.text_model) if rt.text_model else "none")

        # ── 通用上下文层（course_prompt / bot_persona / always_skills / memory / session / now）──
        # include_skills=True：chat 挂 read_skill 工具，急切注入 always-on skill 全文。
        layers = await build_common_context_layers(context, include_skills=True)

        # ── chat agent loop 行为规范（YAML 外部化，agentic_chat.yaml）──
        chat_cfg = load_prompt_dict(_CHAT_PROMPT_PATH)
        loop_system = (chat_cfg.get("loop") or {}).get("system") or ""

        # ── Skill 渐进式揭示：渲染 manifest 清单（always 全文已在 layers.always_skills）──
        from core.skills.skill_service import get_skill_service, render_skills_manifest
        svc = get_skill_service(context.course_id, context.user_id)
        context.skills_manifest = render_skills_manifest(svc.summary_entries())

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

        # ── 工具使用提示 hint（单一数据源，取代手写在 chat.yaml 的工具说明）──
        from core.agent.prompting import build_tool_hint_text
        tool_hint_text = build_tool_hint_text(
            context.enabled_tools,
            context.language,
            skills_manifest=context.skills_manifest,
            extended_tools_manifest=context.extended_tools_manifest,
        )

        # ── 按「稳定性递减」拼装 system prompt（见 assemble_system_prompt docstring）──
        # 通用 6 层取自 layers，chat 专属 4 层（loop_system / skills_manifest / tool_hint /
        # extended）本地组装。assemble_system_prompt 是 prefix-cache 资产，段顺序不动。
        prompt_kwargs = dict(
            loop_system=loop_system,
            course_prompt=layers.course_prompt,
            bot_persona=layers.bot_persona,
            always_skills=layers.always_skills,
            skills_manifest=context.skills_manifest,
            tool_hint_text=tool_hint_text,
            extended_tools_manifest=context.extended_tools_manifest,
            memory_context=layers.memory_context,
            session_summary=layers.session_summary,
            now_text=layers.now_text,
        )
        system_prompt = assemble_system_prompt(**prompt_kwargs)

        log_flow("chat.prompt_assembled",
                 parts=sum(1 for v in prompt_kwargs.values() if v),
                 system_prompt_chars=len(system_prompt),
                 tools_count=len(tool_schemas) if tool_schemas else 0,
                 skills_manifest=bool(context.skills_manifest),
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000))

        # ── 两阶段图片处理（rt.text_model/binding 判断主模型能否看图）──────────
        # 主模型不支持 vision → 视觉模型描述成文字；支持 → 跳过，走 loop 内直注。
        context.user_message = await describe_images(context, context.user_message, rt)

        # cron owner 注入（bot 对话：loop._run_turn 写入 metadata；web 无则不 set，cron 工具不挂载）
        from core.bot.cron_tool import reset_cron_owner, set_cron_owner
        cron_token = set_cron_owner(context.metadata.get("cron_owner"))
        try:
            await run_agent_loop(
                context=context,
                stream=stream,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                client=rt.client,
                model=rt.text_model,
                binding=rt.binding,
            )
        finally:
            reset_cron_owner(cron_token)
            from core.agentic.dynamic_tools import reset_deferred_loader
            reset_deferred_loader(loader_token)


__all__ = ["ChatPipeline"]
