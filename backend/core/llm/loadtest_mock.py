"""压测用 LLM mock（仅 LOAD_TEST_MOCK_LLM=1 时由 core/llm/llm.py 的 _make_call 调用）。

设计要点（对齐 core/agentic/loop.py 的流式解析 + core/llm/reliability.py 的可靠性语义）：

1. mock 返回 async generator，yield 形如 openai.ChatCompletionChunk 的对象。loop.py:184-245
   用 getattr 访问 chunk.choices[0].delta.{content,tool_calls,reasoning_content}，故用
   dataclass 朴素模拟即可，不依赖 openai SDK 的真实类型。

2. maybe_loadtest_mock_stream 必须是 async def（内部 return _gen()），不能本身是 async
   generator。调用链：reliability.with_retry_and_circuit → circuit_breaker.call(_call) →
   `_call` retry 循环里 `await _call_with_image_fallback()` → 它再 `await
   maybe_loadtest_mock_stream()` 拿到 _gen() 这个 async gen 对象。若本函数本身是 async
   gen，`await mock()` 会 TypeError（async generator 不能被 await）。

3. 正常 mock 不抛异常 → reliability 计 success，不污染熔断器、不计 retry（reliability.py:172
   _on_success 在 result 拿到 gen 对象后调用，gen 对象非 Exception 即成功）。

4. H-11 熔断复验：LOAD_TEST_FORCE_FAIL_RATIO 按概率抛 _FakeAPIError(status_code=503)，
   命中 reliability.retryable_status_codes（reliability.py:67-74）→ 触发 retry + 累计
   failure_count → 熔断 OPEN。force_fail 抛在 await 建流阶段，与真实上游建连失败语义一致。

生产零影响：本模块仅被 import（只定义函数，零副作用）；env 不设时 llm.py 的分流 if 永不进入。
"""
from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field


# ---- 用 dataclass 模拟 openai.ChatCompletionChunk 的属性结构（loop.py 用 getattr 访问）----
@dataclass
class _Function:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _ToolCall:
    index: int = 0
    id: str | None = None
    function: _Function = field(default_factory=_Function)


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    reasoning_content: str | None = None


@dataclass
class _Choice:
    delta: _Delta = field(default_factory=_Delta)
    finish_reason: str | None = None


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)


class _FakeAPIError(Exception):
    """模拟上游 5xx：带 status_code 属性，被 reliability 识别为可重试错误。

    reliability.py:301-306 用 getattr(e, 'status_code', None) 提取，503 命中
    retryable_status_codes → 触发指数退避重试；重试耗尽后 _on_failure 累计失败计数。
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


# 模拟答案内容（中文短语，贴近真实 token 粒度；快速排序场景，呼应压测默认提问）
_PANGS = [
    "快速排序", "采用分治", "选取基准", "小于基准", "放左边",
    "大于基准", "放右边", "递归处理", "时间复杂度", "平均 n log n",
    "。", "这是", "压测", "模拟", "回复",
]


def _content_chunk(text: str) -> _Chunk:
    return _Chunk(choices=[_Choice(delta=_Delta(content=text))])


def _final_chunk() -> _Chunk:
    return _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")])


async def _gen(ttft: float, total_chars: int):
    """真 async generator：先 sleep 模拟首 token 延迟，再逐片 yield content，末尾 stop。

    loop.py:213 在首个非空 content 时记录 TTFT，_t_start 取自 create 调用前（loop.py:171），
    故本 sleep 的 ttft 会被计入 ca_llm_first_token_seconds 指标，使压测 TTFT 可控可测。
    """
    if ttft > 0:
        await asyncio.sleep(ttft)
    emitted = 0
    while emitted < total_chars:
        piece = random.choice(_PANGS)
        await asyncio.sleep(0.02)  # 模拟 token 生成间隔（20ms/片）
        yield _content_chunk(piece)
        emitted += len(piece)
    yield _final_chunk()


async def maybe_loadtest_mock_stream(ttft: float, total_chars: int):
    """async def（非 generator！）：被 llm.py 的 _call_with_image_fallback `await` 调用，
    返回 _gen() async generator。

    H-11 熔断复验：LOAD_TEST_FORCE_FAIL_RATIO 按概率抛 503，命中 reliability 重试 + 累计
    失败。抛在 await 阶段（建流时），与真实上游建连失败语义一致。
    """
    ratio_raw = os.getenv("LOAD_TEST_FORCE_FAIL_RATIO", "0") or "0"
    try:
        ratio = float(ratio_raw)
    except ValueError:
        ratio = 0.0
    if ratio > 0 and random.random() < ratio:
        raise _FakeAPIError(503, "Service Unavailable (loadtest force-fail)")
    return _gen(ttft, total_chars)
