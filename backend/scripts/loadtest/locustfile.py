"""Locust 压测脚本：SSE（POST /api/chat）+ WebSocket（/api/run/*）+ 非 LLM 探针。

gevent 兼容性（关键，搞错会死锁）：
  - SSE 用 gevent-patched requests（stream=True + iter_lines）；
  - WS 用 websocket-client（同步库，gevent patch socket 后协程友好）；
  - 绝不用 websockets(asyncio)——会和 Locust 的 gevent loop 冲突死锁。

数据：on_start 读 seed.py 产出的 tokens.json，**不在压测期登录**（/api/auth/login 的
bcrypt 同步校验 CPU-bound，并发登录会挡住 greenlet、把 Locust worker 卡死）。
建议 Locust 用户数 ≤ tokens.json 的 student 数（1:1），避免 token 复用撞 per-token
20/min 限流（api/chat.py:49）。

TTFT / total 分两次 fire（name 后缀 _ttft / _total），让 Grafana 分离「首 token 延迟」
与「整 turn 耗时」——TTFT 是用户体验核心指标，必须独立看。
H-15：收到 error 事件 = 正常响应（got_error），fire success；只有「连接断开且无
error/answer/done」（silence）才 fire failure——避免把 TRM 的 error 事件误判成压测失败。

运行（宿主 Windows，先 pip install -r requirements.loadtest.txt）：
  locust -f backend/scripts/loadtest/locustfile.py --host http://localhost:8000
  浏览器开 http://localhost:8089 → 直接点 Start（OneMinuteShape 自动控制 20 用户 / 1 分钟）。
  --host 要与 tokens.json 的 base_url 一致（self.client 探针用它）。
  覆盖时长：LOADTEST_DURATION_SEC=300 locust ...（默认 60 秒）。
"""
from __future__ import annotations

import json
import os
import random
import time

import requests
import websocket  # websocket-client（同步库；勿用 websockets async）
from locust import HttpUser, between, events, task

TOKENS_FILE = os.getenv("TOKENS_FILE", os.path.join(os.path.dirname(__file__), "tokens.json"))
DEFAULT_QUESTION = "请用一句话解释快速排序的核心思想"
LOADTEST_DURATION_SEC = int(os.getenv("LOADTEST_DURATION_SEC", "60"))
LOADTEST_USERS = int(os.getenv("LOADTEST_USERS", "20"))
LOADTEST_SPAWN_RATE = int(os.getenv("LOADTEST_SPAWN_RATE", "2"))


def _load_tokens() -> dict:
    with open(TOKENS_FILE, encoding="utf-8") as f:
        return json.load(f)


class ChatUser(HttpUser):
    """模拟学生：随机做 chat(SSE) / quiz / solve / research(WS) + health/courses 探针。"""

    wait_time = between(1, 3)

    def on_start(self):
        data = _load_tokens()
        self.base = data["base_url"].rstrip("/")
        self.ws_base = self.base.replace("http://", "ws://").replace("https://", "wss://")
        self.students = data.get("students") or []
        if not self.students:
            raise SystemExit("tokens.json 无 students；先在宿主跑 seed.py 造数据")
        stu = random.choice(self.students)
        self.token = stu["token"]
        self.course_ids = stu.get("course_ids") or []
        self.headers = {"Authorization": f"Bearer {self.token}"}

    # ---------- 非 LLM 探针（用 self.client 自动 fire；验证基础链路不被打挂）----------
    @task(2)
    def health(self):
        self.client.get("/api/health", timeout=5, name="health")

    @task(1)
    def list_courses(self):
        self.client.get("/api/courses", headers=self.headers, timeout=5, name="courses")

    # ---------- SSE：POST /api/chat（LLM 主链路，权重最高）----------
    @task(5)
    def chat_sse(self):
        course_id = random.choice(self.course_ids) if self.course_ids else "general"
        body = {
            "course_id": course_id,
            "message": DEFAULT_QUESTION,
            "chat_mode": "chat",
            "history": [],
            "tools": [],  # 空 tools：跳过 KB/web 工具，纯压 LLM 流式链路（mock 接管 85%）
        }
        t0 = time.time()
        ttft = None
        got_content = got_error = got_done = False
        try:
            with requests.post(f"{self.base}/api/chat", headers=self.headers,
                               json=body, stream=True, timeout=90) as r:
                if r.status_code != 200:
                    events.request.fire(
                        request_type="POST", name="chat_total",
                        response_time=(time.time() - t0) * 1000, response_length=0,
                        exception=RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}"),
                    )
                    return
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    try:
                        evt = json.loads(raw[6:])
                    except json.JSONDecodeError:
                        continue
                    t = evt.get("type")
                    if t in ("token", "answer"):
                        got_content = True
                        if ttft is None:
                            ttft = (time.time() - t0) * 1000
                            events.request.fire(
                                request_type="SSE", name="chat_ttft",
                                response_time=ttft, response_length=0,
                            )
                    if t == "error":
                        got_error = True
                    if t == "done":
                        got_done = True
                        break
            self._finish("chat", t0, got_content, got_error, got_done)
        except Exception as exc:
            events.request.fire(
                request_type="POST", name="chat_total",
                response_time=(time.time() - t0) * 1000, response_length=0, exception=exc,
            )

    # ---------- WebSocket：/api/run/{cap}（token 经 ?token= query param）----------
    @task(3)
    def run_quiz_ws(self):
        self._run_ws("quiz", "请出一道关于快速排序的单选题")

    @task(2)
    def run_solve_ws(self):
        self._run_ws("deep_solve", "帮我分析：快速排序最坏复杂度为什么是 O(n^2)")

    @task(1)
    def run_research_ws(self):
        # deep_research 默认带 rag（若要复验 H-10 实例池 evict，把 tools 改 ["rag"]）
        self._run_ws("deep_research", "调研快速排序与归并排序的对比")

    def _run_ws(self, cap: str, question: str):
        if not self.course_ids:
            return
        course_id = random.choice(self.course_ids)
        url = f"{self.ws_base}/api/run/{cap}?token={self.token}"
        body = {
            "type": "start_turn",
            "course_id": course_id,
            "question": question,
            "language": "zh",
            "tools": [],
            "history": [],
        }
        t0 = time.time()
        ttft = None
        got_content = got_error = got_done = False
        try:
            ws = websocket.create_connection(url, timeout=90)
            ws.send(json.dumps(body))
            while True:
                raw = ws.recv()
                if not raw:
                    break
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t in ("token", "answer", "quiz_question"):
                    got_content = True
                    if ttft is None:
                        ttft = (time.time() - t0) * 1000
                        events.request.fire(
                            request_type="WS", name=f"{cap}_ttft",
                            response_time=ttft, response_length=0,
                        )
                if t == "wait_for_input":
                    # ask_user 工具暂停：投递回复继续 turn（mock+tools=[] 一般不触发，保留兼容）
                    ws.send(json.dumps({"type": "submit_user_reply", "text": "继续"}))
                if t == "error":
                    got_error = True
                if t == "done":
                    got_done = True
                    break
            ws.close()
            self._finish(cap, t0, got_content, got_error, got_done)
        except Exception as exc:
            events.request.fire(
                request_type="WS", name=f"{cap}_total",
                response_time=(time.time() - t0) * 1000, response_length=0, exception=exc,
            )

    def _finish(self, name: str, t0: float, got_content: bool, got_error: bool, got_done: bool):
        """H-15：done/error/content 任一即算成功响应；全无=silence，fire failure。

        mock 模式正常路径：token...→done（got_content+got_done，success）。
        FORCE_FAIL 路径：error 事件（got_error，仍 success——error 是合法响应）。
        异常路径：连接断开且无任何事件 → silence（真正的失败，暴露静默 bug）。
        """
        total_ms = (time.time() - t0) * 1000
        if got_done or got_error or got_content:
            events.request.fire(
                request_type="RUN", name=f"{name}_total",
                response_time=total_ms, response_length=0,
            )
        else:
            events.request.fire(
                request_type="RUN", name=f"{name}_total",
                response_time=total_ms, response_length=0,
                exception=RuntimeError("silence: 断连且无 error/answer/done"),
            )
