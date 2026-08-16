"""打一次与主题图抽取同款的请求，把模型返回体原样打印出来（不是 pytest）。

用法（在仓库根目录）：
  docker exec -w /app course-agent-worker-1 .venv/bin/python -m scripts.dump_topic_extract_llm_shape
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


async def main() -> None:
    from core.llm.llm import client as llm
    from settings import get_settings

    s = get_settings()
    model = s.course_topic.extract_model or s.llm.text_model
    from core.llm.reasoning_params import apply_reasoning_kwargs

    prompt = (
        "章节：实验二 元件的伏安特性\n\n"
        "在元件伏安特性的基础上测量等效内阻。戴维南等效是可单独考核的知识点。\n\n"
        '输出严格 JSON（无 markdown 围栏）：'
        '{"topics": [{"label": "知识点名", "definition": "一句话定义"}], '
        '"edges": [{"src": "前置点", "dst": "后继点", "strength": 0.8, "evidence": "原文句"}]}'
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8000,
        "stream": False,
    }
    apply_reasoning_kwargs(kwargs, s.llm.binding, model, llm)
    resp = await llm.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    dump = {
        "request": {
            "model": model,
            "max_tokens": 8000,
            "extra_body": kwargs.get("extra_body"),
        },
        "finish_reason": choice.finish_reason,
        "usage": _to_dict(getattr(resp, "usage", None)),
        "message_type": type(msg).__name__,
        "message_keys": sorted(
            set(list(getattr(msg, "model_fields", {}) or []) + list(getattr(msg, "model_extra", {}) or []))
        ) or [k for k in dir(msg) if not k.startswith("_")],
        "content": getattr(msg, "content", None),
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "raw_message": _to_dict(msg),
    }
    print(json.dumps(dump, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
