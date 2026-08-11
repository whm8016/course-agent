"""Anthropic → OpenAI chat.completions 接口适配器。

将 anthropic.AsyncAnthropic 的原生 API 包装成与 openai.AsyncOpenAI 一致的
client.chat.completions.create(**kwargs) 接口，使 agent loop 及其他调用方
无需感知底层供应商差异。

设计参考 core/agentic/client.py 中的 _ProviderOpenAIAdapter 和
_ProviderOpenAIStream，核心机制相同：
  - 非流式：直接调用 anthropic.messages.create，将结果封装为 SimpleNamespace
  - 流式：asyncio.Queue 桥接 anthropic 流 → 逐 chunk yield OpenAI 格式
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic tool_call delta 的 index 计数器（流式时每个 tool_use block 对应一个 index）
_SENTINEL = object()  # 队列结束哨兵


def _openai_chunk(
    *,
    content: str | None = None,
    tool_call: Any | None = None,
    index: int = 0,
    finish_reason: str | None = None,
) -> Any:
    """构造与 openai SDK 兼容的流式 chunk SimpleNamespace。"""
    tool_calls = None
    if tool_call is not None:
        fn = SimpleNamespace(
            name=getattr(tool_call, "name", ""),
            arguments=json.dumps(
                getattr(tool_call, "input", {}) or {}, ensure_ascii=False
            ),
        )
        tool_calls = [
            SimpleNamespace(
                index=index,
                id=getattr(tool_call, "id", f"call_{index}"),
                type="function",
                function=fn,
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _openai_usage_chunk(*, input_tokens: int, output_tokens: int, cache_read: int) -> Any:
    """构造与 OpenAI 末块同形态的 usage chunk（choices=[] + usage 字段）。

    成本可观测性用：把 Anthropic 的 message_start(message.usage) + message_delta(usage)
    归一成 OpenAI 形态，使 loop._one_round 的 usage_from_response_chunk 一套代码读两类 provider。
    """
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
        ),
    )


def _convert_tools(
    tools: list[dict[str, Any]] | None, *, cache_control: bool = False
) -> list[dict[str, Any]] | None:
    """将 OpenAI function calling schema 转换为 Anthropic tool schema。

    cache_control=True 时给最后一个工具加 ephemeral 断点——工具 schema 是 T1 静态锚点
    （同会话逐字一致），缓存命中≈0.1x 成本。对标 Anthropic Prompt Caching（断点放 tools→system→messages）。
    """
    if not tools:
        return None
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    if cache_control and result:
        result[-1]["cache_control"] = {"type": "ephemeral"}
    return result


def _convert_messages(
    messages: list[dict[str, Any]], *, cache_control: bool = False
) -> tuple[Any, list[dict[str, Any]]]:
    """拆分 system prompt，将 OpenAI messages 转换为 Anthropic 格式。

    返回 (system, anthropic_messages)。system 在 cache_control=False 时为 str（逐字同旧行为）；
    cache_control=True 时为 text block 列表，并在首个 block（T1 稳定前缀）放 ephemeral 断点——
    _build_messages 已在 T1/T2 边界把 system 拆成两条消息，这里收集后给第一条加断点，T2/KB-seed
    等易变后缀不缓存。Anthropic system 参数两种形态都接受。
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        if role == "tool":
            # OpenAI role=tool → Anthropic role=user + tool_result content block
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content or "",
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # assistant 带 tool_calls → 转为 Anthropic assistant + tool_use blocks
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    inp = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": inp,
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
            continue

        # 普通 user / assistant
        if isinstance(content, list):
            # 多模态 content（image_url 等）→ 转 Anthropic image source block
            converted.append({
                "role": role,
                "content": _convert_content_blocks_to_anthropic(content),
            })
        else:
            converted.append({"role": role, "content": content or ""})

    if cache_control and system_parts:
        # 首个 system block（T1 稳定前缀）放 cache 断点；其余（T2 易变层 / KB-seed）不缓存。
        blocks: list[dict[str, Any]] = []
        for i, part in enumerate(system_parts):
            block = {"type": "text", "text": part}
            if i == 0:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks, converted
    return "\n\n".join(system_parts), converted


def _convert_content_blocks_to_anthropic(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 OpenAI 多模态 content 数组转成 Anthropic content blocks。

    OpenAI:    {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
    Anthropic: {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}

    Anthropic 仅接受 base64 source，不接受 http url 远程图；遇到 http url 本期跳过
    （P2 加下载转 base64）。此前直接原样透传 OpenAI image_url 块会导致 Claude API 400。
    """
    converted: list[dict[str, Any]] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            converted.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                header, _, b64 = url.partition(",")
                media_type = header[len("data:"):-len(";base64")] or "image/png"
                converted.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
            else:
                logger.warning("AnthropicAdapter: 跳过非 data URL 图片 %s", url[:80])
        else:
            # tool_use / tool_result 已由上层分支处理；未知类型原样保留
            converted.append(part)
    return converted


class AnthropicAdapter:
    """将 anthropic.AsyncAnthropic 包装成 OpenAI chat.completions 接口。

    用法与 openai.AsyncOpenAI 相同：
        adapter = AnthropicAdapter(api_key="sk-ant-...")
        response = await adapter.chat.completions.create(
            model="claude-sonnet-4-5",
            messages=[...],
            stream=True,
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url or None,
            )
        except ImportError as exc:
            raise ImportError(
                "anthropic 包未安装。请运行：pip install anthropic"
            ) from exc

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs: Any) -> Any:
        stream: bool = bool(kwargs.pop("stream", False))
        messages: list[dict[str, Any]] = kwargs.pop("messages", [])
        model: str = kwargs.pop("model", "claude-sonnet-4-5")
        tools_openai = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        temperature: float = float(kwargs.pop("temperature", 0.7))
        max_tokens: int = int(
            kwargs.pop("max_completion_tokens", None) or kwargs.pop("max_tokens", 4096)
        )
        kwargs.pop("stream_options", None)

        # cache_control（CONTEXT_BUDGET__CACHE_CONTROL_ENABLED）：开则给 system T1 前缀 + 末工具
        # 加 ephemeral 断点，命中≈0.1x 成本；默认关→system 为裸字符串，逐字节同旧行为。
        from settings import get_settings
        _cc = bool(get_settings().context_budget.cache_control_enabled)
        system, ant_messages = _convert_messages(messages, cache_control=_cc)
        ant_tools = _convert_tools(tools_openai, cache_control=_cc)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": ant_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            create_kwargs["system"] = system
        if ant_tools:
            create_kwargs["tools"] = ant_tools

        if stream:
            return _AnthropicStream(client=self._client, create_kwargs=create_kwargs)

        # 非流式
        resp = await self._client.messages.create(**create_kwargs)
        return _response_to_openai(resp)


def _response_to_openai(resp: Any) -> Any:
    """将 anthropic.types.Message 转换为 OpenAI 非流式 response 格式。"""
    text_parts: list[str] = []
    tool_calls: list[Any] = []

    for idx, block in enumerate(resp.content or []):
        btype = getattr(block, "type", "")
        if btype == "text":
            text_parts.append(getattr(block, "text", ""))
        elif btype == "tool_use":
            fn = SimpleNamespace(
                name=getattr(block, "name", ""),
                arguments=json.dumps(
                    getattr(block, "input", {}) or {}, ensure_ascii=False
                ),
            )
            tool_calls.append(
                SimpleNamespace(
                    index=idx,
                    id=getattr(block, "id", f"call_{idx}"),
                    type="function",
                    function=fn,
                )
            )

    message = SimpleNamespace(
        content="".join(text_parts) or None,
        tool_calls=tool_calls or None,
    )
    finish = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)]
    )


class _AnthropicStream:
    """将 anthropic streaming_messages 转换为 OpenAI 流式 chunk 序列。

    通过 asyncio.Queue 桥接：后台 task 消费 anthropic stream，
    将 OpenAI 格式的 chunk 入队；__aiter__ 逐个出队。
    """

    def __init__(self, *, client: Any, create_kwargs: dict[str, Any]) -> None:
        self._client = client
        self._create_kwargs = create_kwargs
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def __aiter__(self) -> "_AnthropicStream":
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return self

    async def __anext__(self) -> Any:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        try:
            async with self._client.messages.stream(**self._create_kwargs) as stream:
                tool_index = 0
                current_tool: dict[str, Any] | None = None
                current_tool_args = ""
                # 成本采集：Anthropic usage 分散在两事件——message_start 带 input/cache_read，
                # message_delta 带 output。累积后在 message_stop 合成 OpenAI 形态 usage 块。
                usage_in = 0
                usage_cache = 0
                usage_out = 0

                async for event in stream:
                    etype = getattr(event, "type", "")

                    # 文本 delta
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None:
                            continue
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                await self._queue.put(_openai_chunk(content=text))
                        elif dtype == "input_json_delta":
                            # tool_use 的参数片段
                            current_tool_args += getattr(delta, "partial_json", "")

                    elif etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            current_tool = {
                                "id": getattr(block, "id", f"call_{tool_index}"),
                                "name": getattr(block, "name", ""),
                            }
                            current_tool_args = ""

                    elif etype == "content_block_stop":
                        if current_tool is not None:
                            # 发出完整的 tool_call chunk
                            try:
                                inp = json.loads(current_tool_args) if current_tool_args else {}
                            except json.JSONDecodeError:
                                inp = {}
                            tc = SimpleNamespace(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                input=inp,
                            )
                            await self._queue.put(
                                _openai_chunk(tool_call=tc, index=tool_index)
                            )
                            tool_index += 1
                            current_tool = None
                            current_tool_args = ""

                    elif etype == "message_stop":
                        stop_reason = getattr(
                            getattr(event, "message", None), "stop_reason", "stop"
                        ) or "stop"
                        finish = "tool_calls" if tool_index > 0 else stop_reason
                        await self._queue.put(_openai_chunk(finish_reason=finish))
                        # 合成 OpenAI 形态 usage 块（成本采集）——消息正常结束时才攒齐完整 usage
                        await self._queue.put(_openai_usage_chunk(
                            input_tokens=usage_in, output_tokens=usage_out, cache_read=usage_cache,
                        ))

                    elif etype == "message_start":
                        # input_tokens + cache_read 在消息开始时给出
                        _msg = getattr(event, "message", None)
                        _u = getattr(_msg, "usage", None) if _msg else None
                        if _u is not None:
                            usage_in = int(getattr(_u, "input_tokens", 0) or 0)
                            usage_cache = int(getattr(_u, "cache_read_input_tokens", 0) or 0)

                    elif etype == "message_delta":
                        # output_tokens 在消息增量结束时给出（累计值）
                        _u = getattr(event, "usage", None)
                        if _u is not None:
                            usage_out = int(getattr(_u, "output_tokens", 0) or 0)

        except Exception as exc:
            logger.exception("AnthropicStream error")
            await self._queue.put(exc)
        finally:
            await self._queue.put(_SENTINEL)


__all__ = ["AnthropicAdapter"]
