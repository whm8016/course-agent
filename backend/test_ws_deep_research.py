"""
Minimal WebSocket client for Deep Research Step 0.

Run backend first (cwd must be backend):
  python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

Then:
  python test_ws_deep_research.py
  python test_ws_deep_research.py your_course_kb_name

Requires: pip install websockets

kb_name is required for real RAG grounding; without it the pipeline still runs
but RAG is skipped and the report should state insufficient sources.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

try:
    import websockets
except ImportError:
    print("Missing package: pip install websockets")
    sys.exit(1)


WS_URL = "ws://127.0.0.1:8000/api/deep-research/run"

# Optional: set DEEP_RESEARCH_KB_NAME in env or pass as argv[1]
KB_NAME = (sys.argv[1] if len(sys.argv) > 1 else None) or os.getenv("DEEP_RESEARCH_KB_NAME")

START_MESSAGE: dict = {
    "type": "start",
    "topic": "本地调试：RAG 是什么",
    "config": {
        "mode": "notes",
        "depth": "quick",
        "sources": ["kb"],
    },
    "language": "zh",
}

if KB_NAME:
    START_MESSAGE["kb_name"] = KB_NAME
else:
    print("Warning: no kb_name — RAG will be skipped. Pass course id: python test_ws_deep_research.py <kb_name>")


async def main() -> None:
    payload = json.dumps(START_MESSAGE, ensure_ascii=False)
    async with websockets.connect(WS_URL) as ws:
        await ws.send(payload)
        result_data: dict | None = None
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("type") == "result":
                result_data = data
                break
            if data.get("type") == "error":
                sys.exit(1)

    if result_data:
        report = result_data.get("report") or ""
        meta = result_data.get("metadata") or {}
        print("\n--- report preview (first 500 chars) ---")
        print(report[:500])
        if len(report) > 500:
            print("...")
        print("\n--- metadata ---")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print("\nfinal_report_path:", result_data.get("final_report_path", ""))


if __name__ == "__main__":
    asyncio.run(main())
