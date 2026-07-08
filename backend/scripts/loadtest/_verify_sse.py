"""手动 SSE 端到端验证（非压测，单次确认）：用 tokens.json 学生 token 打一个 chat，
打印 TTFT / total / event 类型序列。mock 模式应见 turn_started→token...→done，
TTFT ≈ LOAD_TEST_MOCK_TTFT_MS（默认 600ms）。
用法：./backend/venv/Scripts/python.exe backend/scripts/loadtest/_verify_sse.py [tokens.json]
"""
import json
import sys
import time

import httpx

tokens_file = sys.argv[1] if len(sys.argv) > 1 else "tokens.json"
data = json.load(open(tokens_file, encoding="utf-8"))
stu = data["students"][0]
base = data["base_url"].rstrip("/")
headers = {"Authorization": f"Bearer {stu['token']}"}
body = {
    "course_id": stu["course_ids"][0] if stu["course_ids"] else "general",
    "message": "解释快速排序",
    "chat_mode": "chat",
    "history": [],
    "tools": [],
}
t0 = time.time()
ttft = None
types = []
with httpx.stream("POST", f"{base}/api/chat", headers=headers, json=body, timeout=60) as r:
    print("HTTP", r.status_code)
    if r.status_code != 200:
        print(r.read().decode(errors="replace")[:300])
        sys.exit(1)
    for line in r.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        evt = json.loads(line[6:])
        t = evt.get("type")
        types.append(t)
        if t in ("token", "answer") and ttft is None:
            ttft = (time.time() - t0) * 1000
            print(f"TTFT  = {ttft:.0f} ms  (first '{t}')")
        if t == "done":
            print(f"total = {(time.time() - t0) * 1000:.0f} ms")
            break
print("event types:", types[:20])
