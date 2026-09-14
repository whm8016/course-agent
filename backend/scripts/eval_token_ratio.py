"""token 估算 vs 真实 usage 的 ratio 验收（估算准确性整改的"准了"证明）。

测量口径（与生产主链路同构）：
  - 估算侧：发送前 ``count_tokens_messages(messages, tools=tools)``（本轮整改后的统一估算器）
  - 真实侧：OpenAI 兼容流式请求 ``stream_options={"include_usage": True}``，末块
    ``usage.prompt_tokens``（与 loop._one_round 同一捕获方式）
  - 采样条件：仅取 ``prompt_tokens_details.cached_tokens == 0`` 的样本（prompt caching
    会污染：cache 命中轮的 input_tokens 口径与全量文本估算不可比），首轮天然满足

验收标准（plan 约定）：P50 ∈ [0.85, 1.25] 且无样本 < 0.7（低估才有 413 风险）。

用法：
  真实端点（由用户执行，烧少量真实 token）：
      cd backend && .venv/Scripts/python scripts/eval_token_ratio.py --rounds 20
  mock 自测（不真打 LLM，只验证脚本自身的采样/聚合/判定逻辑）：
      .venv/Scripts/python scripts/eval_token_ratio.py --mock
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台下中文输出防乱码
    sys.stdout.reconfigure(encoding="utf-8")

from core.agentic.context_window import count_tokens_messages  # noqa: E402


# ---------------------------------------------------------------------------
# 样本构造：覆盖生产形态（纯中文/纯英文/带 tool_calls 历史/带图/带 tools 定义）
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "在课程知识库中检索与问题相关的文本块，返回内容与来源引用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quiz_generate",
            "description": "Generate quiz questions for the student based on a requirement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["requirement"],
            },
        },
    },
]


def _tiny_png_data_url() -> str:
    """1x1 红色 PNG 的 data URL（图片 token 走估算器常数 1280，不占真实体积）。"""
    import base64

    # 1x1 红色像素的最小合法 PNG
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
        "53de0000000c4944415408d763f8cfc000000301010018dd8db00000000049"
        "454e44ae426082"
    )
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


def build_round(i: int) -> tuple[list[dict], list[dict] | None]:
    """第 i 轮的 (messages, tools)。轮转四种生产形态。"""
    kind = i % 4
    msgs: list[dict] = [{"role": "system", "content": "你是高校课程助教，用中文答疑。" * 3}]
    if kind == 0:  # 纯中文
        msgs.append({"role": "user", "content": f"第{i}问：请解释牛顿第二定律，并给出一个生活中的例子。"})
    elif kind == 1:  # 纯英文
        msgs.append({"role": "user", "content": f"Round {i}: Explain the time complexity of quicksort and when it degrades."})
    elif kind == 2:  # 带 tool_calls 历史（assistant 调用 + tool 结果）
        msgs += [
            {"role": "user", "content": f"第{i}问：帮我查一下 CourseAgent 项目的索引分布式锁实现。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "rag_search", "arguments": '{"query": "索引 分布式锁 租约 续约"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{i}", "name": "rag_search",
             "content": "60s 短租约 + 20s reacquire 续约 + 不偷锁；extend 默认加法滚雪球坑。" * 20},
        ]
    else:  # 带图（multimodal，图片走常数估算）
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"第{i}问：描述这张图片的内容。"},
                {"type": "image_url", "image_url": {"url": _tiny_png_data_url()}},
            ],
        })
    return msgs, _TOOLS


# ---------------------------------------------------------------------------
# 真实采样 / mock 采样
# ---------------------------------------------------------------------------

async def _real_input_tokens(messages: list[dict], tools: list[dict] | None) -> tuple[int, int]:
    """流式打一发，返回 (real_input_tokens, cache_read_tokens)。cached>0 的样本由上层丢弃。"""
    import openai

    from settings import get_settings

    s = get_settings().llm
    client = openai.AsyncOpenAI(base_url=s.base_url, api_key=s.api_key)
    try:
        kwargs: dict = {
            "model": s.text_model,
            "messages": messages,
            "max_tokens": 32,  # 只验输入侧，输出给最小
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none"  # 只要 token 计数，不让模型真调工具
        usage = None
        async for chunk in await client.chat.completions.create(**kwargs):
            u = getattr(chunk, "usage", None)
            if u is not None and getattr(u, "prompt_tokens", None) is not None:
                usage = u
        if usage is None:
            raise RuntimeError("流末无 usage 块（端点不支持 stream_options？）")
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        return int(usage.prompt_tokens), cached
    finally:
        await client.close()


async def _mock_input_tokens(messages: list[dict], tools: list[dict] | None) -> tuple[int, int]:
    """mock：真实侧=估算侧×固定系数（模拟 cl100k 对目标模型的 ±20% 偏差），cached 恒 0。"""
    est = count_tokens_messages(messages, tools=tools)
    return int(est * 1.1), 0  # 模拟高估 10% 的模型 tokenizer


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run(rounds: int, mock: bool) -> int:
    sampler = _mock_input_tokens if mock else _real_input_tokens
    label = "MOCK（逻辑自测，不代表估算精度）" if mock else "真实端点"
    print(f"== token 估算 ratio 验收（{label}，目标 P50∈[0.85,1.25] 且无样本<0.7）==")

    ratios: list[float] = []
    for i in range(rounds):
        msgs, tools = build_round(i)
        est = count_tokens_messages(msgs, tools=tools)
        try:
            real, cached = await sampler(msgs, tools)
        except Exception as exc:
            print(f"[{i+1:2d}/{rounds}] 采样失败：{exc}")
            continue
        if cached > 0:
            print(f"[{i+1:2d}/{rounds}] cache_read={cached}>0，丢弃（采样条件：首轮无缓存）")
            continue
        ratio = est / real if real else float("inf")
        ratios.append(ratio)
        print(f"[{i+1:2d}/{rounds}] 形态{i % 4} est={est:6d} real={real:6d} ratio={ratio:.3f}")

    if not ratios:
        print("无有效样本")
        return 2
    p50 = statistics.median(ratios)
    lo, hi = min(ratios), max(ratios)
    ok = 0.85 <= p50 <= 1.25 and lo >= 0.7
    print(f"\n样本 n={len(ratios)}  P50={p50:.3f}  min={lo:.3f}  max={hi:.3f}")
    print("结论：" + ("PASS（达标）" if ok else "FAIL（未达标：低估样本有 413 风险 / 偏差过大）"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=20, help="采样回合数（默认 20，覆盖四种形态）")
    ap.add_argument("--mock", action="store_true", help="不真打 LLM，只验证脚本自身逻辑")
    args = ap.parse_args()
    return asyncio.run(run(args.rounds, args.mock))


if __name__ == "__main__":
    raise SystemExit(main())
