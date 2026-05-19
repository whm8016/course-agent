"""
连续 3 轮对话 + 每轮后查看 memory 是否变化（第 3 轮结束应触发 LLM 重写）。

面向 docker compose：无需在本机再开 uvicorn，只要容器在跑即可。

在项目根目录执行（推荐，在 backend 容器里跑，走容器内 8002）：

  docker compose exec backend python test_memory_three_turns.py --user 用户名 --password 密码

或在宿主机用 Python（需 pip install httpx，连映射端口 8002）：

  python backend/test_memory_three_turns.py --user 用户名 --password 密码

其它示例：

  docker compose exec backend python test_memory_three_turns.py -u 用户名 -p 密码 --clear
  docker compose exec backend python test_memory_three_turns.py -u 用户名 -p 密码 --endpoint lightrag --course circuit_analysis

环境变量（可选）：
  TEST_BASE_URL   默认 http://127.0.0.1:8002（与 docker-compose 端口一致）
  TEST_USERNAME / TEST_PASSWORD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import httpx
except ImportError:
    print("请先安装: pip install httpx")
    sys.exit(1)

DEFAULT_QUESTIONS = [
    "用一句话说明 RC 积分电路是做什么的。",
    "实验七和实验三在课程里通常分别讲什么？各用一句话。",
    "逐步给我讲解滑线电阻在实验电路里常见的接法。",
]

ENDPOINTS = {
    "chat": "/api/chat",
    "lightrag": "/api/chat/lightrag",
}


def _login(client: httpx.Client, base: str, username: str, password: str) -> str:
    r = client.post(
        f"{base}/api/auth/login",
        json={"username": username, "password": password},
        timeout=30.0,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"登录响应无 token: {data}")
    return token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _get_memory(client: httpx.Client, base: str, token: str) -> dict:
    r = client.get(f"{base}/api/memory", headers=_headers(token), timeout=30.0)
    r.raise_for_status()
    return r.json()


def _clear_memory(client: httpx.Client, base: str, token: str) -> None:
    r = client.post(
        f"{base}/api/memory/clear",
        headers=_headers(token),
        json={},
        timeout=30.0,
    )
    r.raise_for_status()


def _print_memory(label: str, snap: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"【{label}】")
    print(f"{'─' * 60}")
    summary = (snap.get("summary") or "").strip()
    profile = (snap.get("profile") or "").strip()
    print("--- summary ---")
    print(summary if summary else "(empty)")
    print("--- profile ---")
    print(profile if profile else "(empty)")


def _memory_fingerprint(snap: dict) -> tuple[str, str]:
    return (
        (snap.get("summary") or "").strip(),
        (snap.get("profile") or "").strip(),
    )


def _chat_sse(
    client: httpx.Client,
    base: str,
    token: str,
    path: str,
    *,
    course_id: str,
    message: str,
    history: list[dict],
    session_id: str | None,
) -> tuple[str, dict]:
    """消费 SSE，返回 (完整回答, done 的 metadata)。"""
    body: dict = {
        "course_id": course_id,
        "message": message,
        "history": history,
        "chat_mode": "chat",
    }
    if session_id:
        body["session_id"] = session_id

    answer_parts: list[str] = []
    metadata: dict = {}

    with client.stream(
        "POST",
        f"{base}{path}",
        headers={**_headers(token), "Accept": "text/event-stream"},
        json=body,
        timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "answer":
                answer_parts.append(str(event.get("content") or ""))
            elif etype == "done":
                metadata = event.get("metadata") or {}
            elif etype == "error":
                raise RuntimeError(event.get("content") or "stream error")

    return "".join(answer_parts), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="3 轮对话后观察 learner memory 变化")
    parser.add_argument(
        "--base-url",
        default=os.getenv("TEST_BASE_URL", "http://127.0.0.1:8002"),
        help="Docker 映射端口为 8002（见 docker-compose.yml）",
    )
    parser.add_argument("-u", "--user", default=os.getenv("TEST_USERNAME", ""))
    parser.add_argument("-p", "--password", default=os.getenv("TEST_PASSWORD", ""))
    parser.add_argument(
        "--endpoint",
        choices=list(ENDPOINTS),
        default="lightrag",
        help="chat=/api/chat  lightrag=/api/chat/lightrag（默认）",
    )
    parser.add_argument("--course", default="circuit_analysis")
    parser.add_argument("--clear", action="store_true", help="开始前清空 memory")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        dest="questions",
        help="自定义问题，可多次 -q；不足 3 条时用默认问题补齐",
    )
    args = parser.parse_args()

    username = args.user.strip()
    password = args.password
    if not username or not password:
        print("请提供登录账号：--user / --password 或环境变量 TEST_USERNAME / TEST_PASSWORD")
        sys.exit(1)

    base = args.base_url.rstrip("/")
    path = ENDPOINTS[args.endpoint]
    questions = list(args.questions or [])
    while len(questions) < 3:
        questions.append(DEFAULT_QUESTIONS[len(questions)])

    print(f"BASE_URL={base}")
    print(f"endpoint={path}  course={args.course}")
    print("说明：memory 每累计 3 轮对话后触发 LLM 重写（第 3 轮 done 之后应看到 summary/profile 可能变化）\n")

    with httpx.Client() as client:
        # health
        try:
            h = client.get(f"{base}/api/health", timeout=10.0)
            print(f"health: {h.status_code} {h.text[:120]}")
        except httpx.HTTPError as e:
            print(f"无法连接后端 {base}: {e}")
            sys.exit(1)

        token = _login(client, base, username, password)
        print("登录成功\n")

        if args.clear:
            _clear_memory(client, base, token)
            print("已清空 memory\n")

        prev_fp = _memory_fingerprint(_get_memory(client, base, token))
        _print_memory("初始 memory", {"summary": prev_fp[0], "profile": prev_fp[1]})

        history: list[dict] = []
        session_id: str | None = None

        for i, q in enumerate(questions[:3], start=1):
            print(f"\n{'=' * 60}")
            print(f"第 {i}/3 轮提问: {q}")
            print(f"{'=' * 60}")

            t0 = time.perf_counter()
            try:
                answer, meta = _chat_sse(
                    client,
                    base,
                    token,
                    path,
                    course_id=args.course,
                    message=q,
                    history=history,
                    session_id=session_id,
                )
            except httpx.HTTPStatusError as e:
                print(f"HTTP 错误: {e.response.status_code} {e.response.text[:500]}")
                sys.exit(1)
            except Exception as e:
                print(f"对话失败: {e}")
                sys.exit(1)

            elapsed = time.perf_counter() - t0
            print(f"\n助手回答长度: {len(answer)} 字符, 耗时 {elapsed:.1f}s")
            print(f"回答预览: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            if meta:
                print(f"metadata: {json.dumps(meta, ensure_ascii=False)[:200]}")

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})

            snap = _get_memory(client, base, token)
            fp = _memory_fingerprint(snap)
            changed = fp != prev_fp
            _print_memory(
                f"第 {i} 轮结束后 memory" + (" ← 有变化" if changed else " ← 无变化"),
                snap,
            )
            if i == 3 and not changed:
                print(
                    "\n⚠ 第 3 轮后 memory 仍无变化：检查后端日志是否有 "
                    "'memory refreshed' / 'memory rewrite NO_CHANGE'，"
                    "以及 chat 路径是否已用 += 累加 answer。"
                )
            elif i == 3 and changed:
                print("\n✓ 第 3 轮后 memory 已更新（符合每 3 轮重写一次的设计）。")

            prev_fp = fp

    print("\n全部完成。")


if __name__ == "__main__":
    main()
