"""tool_calls 驱动的 Agent Loop — Web Chat 共享调度内核。

一次用户回合 = 一次 run_agent_loop() 调用：

  for iteration in range(max_iterations):
      result = await _one_round(messages, tool_schemas, model)   # 一次 LLM 调用
      if result.has_tool_calls:
          发送 thinking 旁白文本（工具调用前的说明）
          并行分发工具 → 追加 role=tool 消息 → 继续下一轮
      else:
          将内容拆成 token 事件发送 → 发 answer → break

  若轮次预算耗尽 → 禁用工具强制再跑一轮 finish

【为什么选 tool_calls 而非标签协议】
OpenAI 兼容 API 通过 tool_calls 字段（而非内容首行标签）来表达工具调用意图。
这样 prompt 更简洁，且与 GPT / Qwen / DeepSeek 等主流模型的训练方式一致。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from config import LLM_BINDING, TEXT_MODEL
from core.agentic.tool_dispatch import dispatch_tool_calls
from core.agentic.types import DispatchOutcome, LoopOutcome, RoundResult, ToolCall
from core.context import UnifiedContext
from core.llm.llm import _create_with_image_fallback, client as _default_client
from core.llm.multimodal import prepare_multimodal_messages
from core.observability import log_flow
from core.observability.metrics import observe_llm_round
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10
_TOKEN_CHUNK_SIZE = 8  # 模拟流式输出时每个 token 事件携带的字符数


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------
def _snip_tool_results(messages: list[dict[str, Any]], budget_chars: int = 80_000) -> None:
    """Loop 内裁切：当 messages 总字符数超限时，把最早的 role=tool 消息替换为占位符。

    只裁 tool 消息，不动 user/assistant/system，避免丢失对话语义。
    """
    _MARKER = "[工具结果已折叠以节省上下文]"
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= budget_chars:
        return
    for msg in messages:
        if total <= budget_chars:
            break
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content", ""))
        if content == _MARKER:
            continue
        total -= len(content)
        msg["content"] = _MARKER
        total += len(_MARKER)
def _build_messages(system_prompt: str, context: UnifiedContext) -> list[dict[str, Any]]:
    """组装 system prompt + 历史对话 + 当前用户消息为 OpenAI messages 列表。

    图片注入走两步式（对标 DeepTutor 三层解耦）：先拼纯文本 messages，再
    prepare_multimodal_messages 注入附件图片。附件来源合并：attachments 优先，
    回退旧 image_path 单图字段（向后兼容）。
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in (context.conversation_history or []):
        role = msg.get("role", "user")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            # content 可能是 str 或已注入图片的 list（多模态历史），原样透传
            messages.append({"role": role, "content": content})

    # 文档附件文字提取：FILE/PDF 类型按扩展名识别，转文本后拼入用户消息
    user_message = context.user_message
    doc_attachments = [a for a in (context.attachments or []) if not a.is_image() and a.base64]
    if doc_attachments:
        from utils.document_extractor import extract_documents_from_records
        doc_texts, _ = extract_documents_from_records(
            [a.model_dump() for a in doc_attachments]
        )
        # 文档文本已提取进 user_message，释放 base64 省内存（大文档 base64 ~1.33×）
        for a in doc_attachments:
            a.base64 = None
        if doc_texts:
            user_message = (user_message + "\n\n" + "\n\n".join(doc_texts)).strip()

    messages.append({"role": "user", "content": user_message})

    if context.attachments:
        prepare_multimodal_messages(
            messages,
            context.attachments,
            binding=LLM_BINDING,
        )

    return messages


def _get_tool_schemas(context: UnifiedContext) -> list[dict[str, Any]] | None:
    """根据 context.enabled_tools 过滤并返回对应的 OpenAI tool schemas（registry 单一数据源）。"""
    if not context.enabled_tools:
        return None
    from core.agent.registry import get_tool_registry
    schemas = get_tool_registry().schemas_for(context.enabled_tools)
    return schemas or None


async def _one_round(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None,
    model: str,
    llm_client: Any = None,
    live_sink: StreamBus | None = None,
    reasoning_sink: StreamBus | None = None,
    reasoning_stage: str | None = None,
) -> RoundResult:
    """执行一次流式 LLM 调用，返回累积的 content 和 tool_calls。

    默认全程缓冲，由调用方决定如何路由内容：
    - 有工具调用 → content 作为旁白发送 thinking 事件
    - 无工具调用 → content 作为最终答案拆成 token 事件

    传入 live_sink 时，content chunk 边收边透传给前端（真流式）。仅用于最终答案轮
    （该轮 tools=None，LLM 不可能输出 tool_calls，可安全地 chunk-by-chunk 直发，
    首字延迟 ≈ 首 token 生成时间）。content 仍同步累积到 content_parts 供返回。

    llm_client 为 None 时使用全局 _default_client（向后兼容）。
    getattr 防御性访问确保跨供应商 delta 字段缺失时不报 AttributeError。

    reasoning_sink/reasoning_stage 非空时，捕获 delta.reasoning_content（推理模型的
    思考过程）流式发送 thinking 事件（start→thinking_chunk×N→complete），让前端展示
    推理过程。非推理模型无 reasoning_content，不发任何事件，零副作用。所有 reasoning
    emit 失败均被吞掉（仅记日志），绝不打断主答案链路。
    """
    _client = llm_client or _default_client

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas
        kwargs["tool_choice"] = "auto"

    content_parts: list[str] = []
    reasoning_parts: list[str] = []      # 本轮推理（reasoning_content）分片累积
    tc_acc: dict[int, dict[str, Any]] = {}  # 按 index 累积 tool_call 分片
    _t_start = time.perf_counter()
    _ttft_ms: int | None = None
    _reasoning_started = False           # 是否已发出本轮 reasoning 的 start 事件

    stream = await _create_with_image_fallback(_client, kwargs, LLM_BINDING, model)
    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        # 推理模型（如 deepseek-v4-pro）的思考过程：流式发送给前端展示
        rc = getattr(delta, "reasoning_content", None)
        if rc and reasoning_sink is not None and reasoning_stage is not None:
            if not _reasoning_started:
                _reasoning_started = True
                try:
                    await reasoning_sink.emit({
                        "type": "thinking", "stage": reasoning_stage,
                        "call_state": "running", "content": "",
                    })
                except Exception:
                    logger.exception("AgentLoop: reasoning start emit 失败")
            reasoning_parts.append(rc)
            try:
                await reasoning_sink.emit({
                    "type": "thinking_chunk", "stage": reasoning_stage,
                    "content": rc,
                })
            except Exception:
                logger.exception("AgentLoop: reasoning_chunk emit 失败")
        content = getattr(delta, "content", None)
        if content:
            if _ttft_ms is None:
                _ttft_ms = int((time.perf_counter() - _t_start) * 1000)
            content_parts.append(content)
            if live_sink is not None:
                # 真流式：边收 chunk 边透传给前端（仅最终答案轮启用）
                await live_sink.emit({"type": "token", "content": content})
        for tc in getattr(delta, "tool_calls", None) or []:
            if _ttft_ms is None:
                _ttft_ms = int((time.perf_counter() - _t_start) * 1000)
            index = int(getattr(tc, "index", 0) or 0)
            if index not in tc_acc:
                tc_acc[index] = {"id": "", "name": "", "arguments": ""}
            entry = tc_acc[index]
            tcid = getattr(tc, "id", None)
            if tcid:
                entry["id"] += str(tcid)
            fn = getattr(tc, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
                arguments = getattr(fn, "arguments", None)
                if name:
                    entry["name"] += str(name)
                if arguments:
                    entry["arguments"] += str(arguments)

    # 本轮推理收尾：仅在确实产生过 reasoning_content 时发 complete，让前端折叠思考块
    if _reasoning_started and reasoning_sink is not None:
        try:
            await reasoning_sink.emit({
                "type": "thinking", "stage": reasoning_stage,
                "call_state": "complete", "content": "",
            })
        except Exception:
            logger.exception("AgentLoop: reasoning complete emit 失败")

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)

    # 将累积的 tool_call 分片解析为 ToolCall 列表
    tool_calls: list[ToolCall] = []
    for idx in sorted(tc_acc.keys()):
        raw = tc_acc[idx]
        try:
            args = json.loads(raw["arguments"]) if raw["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(
            id=raw["id"],
            name=raw["name"],
            arguments=args,
            arguments_str=raw["arguments"],
        ))

    _elapsed_ms = int((time.perf_counter() - _t_start) * 1000)
    log_flow(
        "agent_loop.llm_round",
        model=model,
        ttft_ms=_ttft_ms,
        elapsed_ms=_elapsed_ms,
        has_tool_calls=bool(tool_calls),
        tool_count=len(tool_calls),
        content_chars=len(content),
        reasoning_chars=len(reasoning),
    )
    return RoundResult(
        content=content,
        tool_calls=tool_calls,
        streamed_live=live_sink is not None,
        elapsed_ms=_elapsed_ms,
        ttft_ms=_ttft_ms,
        reasoning=reasoning,
    )


def _emit_as_tokens(text: str) -> list[dict[str, Any]]:
    """将文本切分为固定大小的 token 事件列表（模拟流式输出，内存操作极快）。"""
    return [
        {"type": "token", "content": text[i: i + _TOKEN_CHUNK_SIZE]}
        for i in range(0, max(1, len(text)), _TOKEN_CHUNK_SIZE)
    ]


def _format_reply(raw_reply: dict[str, Any]) -> str:
    """将 submit_user_reply payload 格式化为 role=tool content 文本，供 LLM 读取。"""
    parts: list[str] = []
    text = raw_reply.get("text")
    if text:
        parts.append(str(text))
    answers = raw_reply.get("answers")
    if isinstance(answers, list):
        for a in answers:
            qid = a.get("questionId", "")
            atxt = a.get("text", "")
            parts.append(f"[{qid}] {atxt}" if qid else str(atxt))
    return "User answered: " + "; ".join(parts) if parts else "User skipped."


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

async def run_agent_loop(
    *,
    context: UnifiedContext,
    stream: StreamBus,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]] | None = None,
    model: str | None = None,
    client: Any | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LoopOutcome:
    """执行 tool_calls 驱动的 Agent Loop，完成一个用户回合。

    参数：
        context:        统一上下文（用户消息、历史、启用工具等）。
        stream:         StreamBus 实例，用于向前端发送事件。
        system_prompt:  已组装好的本轮系统提示词。
        tool_schemas:   传给 LLM 的 OpenAI tool schemas；传 None 表示禁用工具。
                        可用 _get_tool_schemas(context) 从 context.enabled_tools 派生。
        model:          模型名称覆盖；默认使用 config.TEXT_MODEL。
        client:         LLM client（AsyncOpenAI / AnthropicAdapter / AsyncAzureOpenAI）。
                        传 None 时使用全局 _default_client（向后兼容）。
        max_iterations: 单回合最大 LLM 调用次数。最后一次调用强制禁用工具，
                        确保模型输出文字答案。

    向 stream 发送的事件：
        {"type": "thinking",   "content": "..."}   工具调用前的旁白
        {"type": "tool_call",  "tool": ..., "input": ...}
        {"type": "tool_result","tool": ..., "content": ...}
        {"type": "token",      "content": "..."}   最终答案分片
        {"type": "answer",     "content": "..."}   完整最终答案
        {"type": "done",       "metadata": {...}}  本轮结束元数据

    返回：
        LoopOutcome（final_text、rounds、tools_used、completed）。
    """
    # 始终用 chat 主模型（对标 DeepTutor）；图片乐观注入，模型不支持时 Stage-2 降级剥图
    model = model or TEXT_MODEL
    messages = _build_messages(system_prompt, context)
    tools_used: list[str] = []
    final_text = ""
    iteration = 0
    seen_tool_result = False  # 是否已触发过工具调用（决定下一轮 reasoning 的 stage）
    _loop_t0 = time.perf_counter()

    log_flow("agent_loop.start", mode=context.mode, model=model,
             tools_enabled=bool(tool_schemas),
             tool_count=len(tool_schemas) if tool_schemas else 0,
             max_iterations=max_iterations)

    for iteration in range(max_iterations):
        # 最后一轮：禁用工具，强制模型输出文字答案
        is_final_round = iteration == max_iterations - 1
        schemas = None if is_final_round else tool_schemas
        # 防止多轮工具调用后 messages 总量超模型 context window
        _snip_tool_results(messages)
        # 本轮 reasoning 的 stage：首轮=分析问题(thinking)，工具后=整理证据(observing)。
        # 用 stage 区分，避免前端按 stage 分组时同 stage 多轮互相覆盖。
        reasoning_stage = "observing" if seen_tool_result else "thinking"
        log_flow("agent_loop.round", iteration=iteration,
                 is_final_round=is_final_round,
                 tools_active=bool(schemas))

        try:
            # 最后一轮 tools=None，LLM 不可能输出 tool_calls，
            # 可安全地 chunk-by-chunk 真流式透传（首字延迟 ≈ 首 token 时间）
            result = await _one_round(
                messages, schemas, model, client,
                live_sink=stream if is_final_round else None,
                reasoning_sink=stream,
                reasoning_stage=reasoning_stage,
            )
            observe_llm_round(
                mode=context.mode or "chat",
                elapsed_ms=result.elapsed_ms,
                ttft_ms=result.ttft_ms,
            )
        except Exception:
            # 兜底：本轮若已发出 reasoning start（running），补发 complete 避免前端思考块卡转圈
            try:
                await stream.emit({
                    "type": "thinking", "stage": reasoning_stage,
                    "call_state": "complete", "content": "",
                })
            except Exception:
                pass
            logger.exception("AgentLoop: 第 %d 轮 LLM 调用失败", iteration)
            log_flow("agent_loop.llm_error", level=logging.ERROR,
                     iteration=iteration, model=model)
            if iteration == 0:
                raise  # 首轮失败直接向上抛，让 capability 处理错误响应
            final_text = "（抱歉，AI 服务临时异常，请稍后重试。）"
            break

        if result.has_tool_calls:
            tool_names = [tc.name for tc in result.tool_calls]
            log_flow("agent_loop.tool_calls", iteration=iteration,
                     tool_names=tool_names, count=len(tool_names))
            # 工具调用前的旁白文本（通常为空，部分模型会有简短说明）
            if result.content.strip():
                await stream.emit({"type": "thinking", "content": result.content})

            # 将 assistant 消息（含 tool_calls）追加到对话历史
            messages.append({
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments_str},
                    }
                    for tc in result.tool_calls
                ],
            })

            # 并行执行所有工具，发送事件，收集 role=tool 消息
            dispatch: DispatchOutcome = await dispatch_tool_calls(
                result.tool_calls,
                course_id=context.course_id,
                user_id=context.user_id,
                enabled_tools=context.enabled_tools,
                stream=stream,
            )
            tools_used.extend(dispatch.tools_used)
            seen_tool_result = True  # 下一轮 reasoning 归入「整理证据」stage

            if dispatch.pause:
                # ask_user 工具触发暂停：通知前端渲染问题卡片，然后挂起等待用户回复
                waiter = context.metadata.get("wait_for_user_reply")
                if not callable(waiter):
                    # 非 WS 入口（如 HTTP SSE）无法双向交互，直接结束 loop
                    logger.info("AgentLoop: ask_user pause 但无 waiter（非 WS 入口），结束 loop")
                    messages.extend(dispatch.tool_messages)
                    break

                await stream.emit({"type": "ask_user_card", **(dispatch.pause_payload or {})})
                raw_reply = await waiter()
                if raw_reply is None:
                    # waiter 返回 None 表示 turn 被取消
                    messages.extend(dispatch.tool_messages)
                    break

                # 把用户回答写回 pause 那条 role=tool 消息，让 LLM 下一轮能读到
                directive = _format_reply(raw_reply)
                for tm in dispatch.tool_messages:
                    if tm.get("tool_call_id") == dispatch.pause_tool_call_id:
                        tm["content"] = directive
                        break
                messages.extend(dispatch.tool_messages)
                # continue 进入下一轮 LLM，无需 break
            else:
                messages.extend(dispatch.tool_messages)

        else:
            # 无工具调用 → 本轮为最终答案轮
            final_text = result.content
            # 若本轮已通过 live_sink 真流式透传（最后一轮），无需再补发；
            # 否则（非最后一轮的提前收尾）回退到切片模拟流式
            if not result.streamed_live:
                for event in _emit_as_tokens(final_text):
                    await stream.emit(event)
            break

    log_flow("agent_loop.done",
             iterations=iteration + 1,
             tools_used=list(dict.fromkeys(tools_used)),
             answer_chars=len(final_text),
             elapsed_ms=int((time.perf_counter() - _loop_t0) * 1000))

    await stream.emit({"type": "answer", "content": final_text})
    await stream.emit({
        "type": "done",
        "metadata": {
            "mode": context.mode,
            "tools_used": list(dict.fromkeys(tools_used)),  # 去重保序
            "iterations": iteration + 1,
        },
    })

    return LoopOutcome(
        final_text=final_text,
        rounds=iteration + 1,
        tools_used=list(dict.fromkeys(tools_used)),
        completed=True,
    )
