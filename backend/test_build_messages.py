"""
例程：测试 core.llm.llm._build_messages

在 backend 目录下运行：
  python test_build_messages.py
  python test_build_messages.py path/to/your/image.png
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# 1x1 透明 PNG（用于无本地图片时的占位测试）
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001000d0a2db40000000049454e44ae426082"
)


def _preview_messages(messages: list[dict]) -> list[dict]:
    """把 base64 data URL 截短，方便打印。"""
    out = []
    for msg in messages:
        item = {"role": msg["role"]}
        content = msg["content"]
        if isinstance(content, list):
            item["content"] = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    item["content"].append({
                        "type": "image_url",
                        "image_url": {"url": url[:60] + "...(truncated)"},
                    })
                else:
                    item["content"].append(part)
        else:
            item["content"] = content
        out.append(item)
    return out


def main() -> None:
    from core.llm.llm import _build_messages

    system = "你是课程助教，用简洁中文回答。"
    history = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成。"},
    ]

    print("=== 1. 纯文本 ===")
    msgs = _build_messages(system, history, "再举一个例子")
    print(json.dumps(_preview_messages(msgs), ensure_ascii=False, indent=2))
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "再举一个例子"
    assert len(msgs) == 4  # system + 2 history + user

    print("\n=== 2. 带图片 + 自定义问题 ===")
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        tmp = Path(tempfile.gettempdir()) / "test_build_messages.png"
        tmp.write_bytes(_TINY_PNG)
        image_path = str(tmp)
        print(f"(未传图片路径，使用临时文件: {image_path})")

    msgs = _build_messages(system, history, "这张图上写了什么？", image_path=image_path)
    print(json.dumps(_preview_messages(msgs), ensure_ascii=False, indent=2))
    last = msgs[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert last["content"][0]["type"] == "image_url"
    assert last["content"][0]["image_url"]["url"].startswith("data:image/")
    assert last["content"][1]["text"] == "这张图上写了什么？"

    print("\n=== 3. 带图片 + 空 user_message（应走默认文案）===")
    msgs = _build_messages(system, [], user_message="", image_path=image_path)
    assert msgs[-1]["content"][1]["text"] == "请赏析这张邮票图片。"
    print("默认文案:", msgs[-1]["content"][1]["text"])

    print("\n全部断言通过。")


if __name__ == "__main__":
    main()
