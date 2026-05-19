"""
例程：直观感受 core.llm.llm.chat_stream 的流式输出。

在 backend 目录下运行（需 backend/.env 里配置好 DASHSCOPE_API_KEY）：

  python test_chat_stream.py
  python test_chat_stream.py path/to/image.png

终端会边收 token 边打印，像打字机一样。
"""
from __future__ import annotations

import asyncio
import sys
import time


async def _run_stream(
    label: str,
    *,
    system_prompt: str,
    history: list[dict],
    user_message: str,
    image_path: str | None = None,
) -> None:
    from core.llm.llm import chat_stream

    print(f"\n{'=' * 60}")
    print(f"【{label}】")
    print(f"  system: {system_prompt[:40]}...")
    print(f"  history: {len(history)} 条")
    print(f"  user: {user_message!r}")
    if image_path:
        print(f"  image: {image_path}")
    print(f"{'=' * 60}")
    print("助手: ", end="", flush=True)

    t0 = time.perf_counter()
    n_chunks = 0
    async for token in chat_stream(
        system_prompt,
        history,
        user_message,
        image_path=image_path,
    ):
        print(token, end="", flush=True)
        n_chunks += 1

    elapsed = time.perf_counter() - t0
    print()  # 换行
    print(f"\n--- 结束: {n_chunks} 个 chunk, 耗时 {elapsed:.2f}s ---")


async def main() -> None:
    system = "你是课程助教，用简洁中文回答，每次不超过 80 字。"
    history = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成：先查资料，再让模型回答。"},
    ]

    await _run_stream(
        "纯文本流式",
        system_prompt=system,
        history=history,
        user_message="用一句话再解释 RAG，并举个课程场景的例子。",
    )

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path:
        await _run_stream(
            "多模态流式（带图）",
            system_prompt=system,
            history=[],
            user_message="用两三句话描述这张图。",
            image_path=image_path,
        )
    else:
        print("\n（跳过图片测试；若要试视觉模型，请: python test_chat_stream.py <图片路径>）")


if __name__ == "__main__":
    asyncio.run(main())
