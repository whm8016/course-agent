"""
并发压测脚本 —— 对比不同并发度下 /api/chat SSE 接口的真实表现。

用法:
    # 默认并发档位 [1, 5, 10, 20]，每档至少 3 次请求
    python stress_test.py

    # 自定义
    python stress_test.py --concurrency 20 50 100 --per-level 200 --users 300 --base-url http://localhost:8002

注意:
    /api/chat/lightrag 有 slowapi 限流 20/min（按 IP），压测前建议临时放宽或注释掉。
    脚本会自动注册一批临时用户来构建用户池。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid

import aiohttp

# ── 默认参数 ──────────────────────────────────────────────────
DEFAULT_BASE = "http://localhost:8002"
DEFAULT_CONCURRENCY = [1, 5, 10, 20]
DEFAULT_PER_LEVEL = 3
DEFAULT_USERS = 100
DEFAULT_COURSE = "algorithm"
DEFAULT_MSG = "什么是快速排序？请简要说明原理和时间复杂度。"
DEFAULT_HISTORY_TURNS = 1
DEFAULT_MESSAGES = [
    "什么是快速排序？请简要说明原理和时间复杂度。",
    "归并排序和快速排序的主要差异是什么？",
    "二叉搜索树在最坏情况下复杂度是多少？",
    "请用一步一步方式解释 Dijkstra 算法。",
    "给我 3 道算法复杂度相关练习题。",
]


# ── 工具 ──────────────────────────────────────────────────────
async def register_temp_user(base: str) -> str:
    name = f"bench_{uuid.uuid4().hex[:8]}"
    payload = {"username": name, "password": "bench1234", "display_name": "压测用户"}
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{base}/api/auth/register", json=payload) as r:
            body = await r.json()
            if r.status == 409:
                async with s.post(
                    f"{base}/api/auth/login",
                    json={"username": name, "password": "bench1234"},
                ) as r2:
                    body = await r2.json()
            return body["token"]


async def register_user_pool(base: str, users: int, parallel: int = 20) -> list[str]:
    """预注册一批用户并拿到 token，模拟真实多用户场景。"""
    sem = asyncio.Semaphore(max(1, parallel))
    tokens: list[str] = []

    async def _one_user() -> str:
        async with sem:
            return await register_temp_user(base)

    results = await asyncio.gather(*(_one_user() for _ in range(users)), return_exceptions=True)
    for item in results:
        if isinstance(item, Exception):
            continue
        tokens.append(item)
    return tokens


def build_history(turns: int) -> list[dict]:
    if turns <= 0:
        return []
    samples = [
        ("user", "老师我基础比较薄弱，能从入门讲吗？"),
        ("assistant", "可以，我们先从概念和例子开始。"),
        ("user", "时间复杂度和空间复杂度怎么区分？"),
        ("assistant", "时间复杂度看运行步数增长，空间复杂度看额外内存增长。"),
    ]
    # 历史长度按“对话轮次”计算，每轮含 user + assistant 两条消息。
    max_msgs = min(len(samples), turns * 2)
    chosen = samples[:max_msgs]
    return [{"role": role, "content": content} for role, content in chosen]


async def send_chat_sse(
    session: aiohttp.ClientSession,
    base: str,
    token: str,
    payload: dict,
) -> dict:
    """发一次 /api/chat SSE 请求，返回计时结果。"""
    headers = {"Authorization": f"Bearer {token}"}

    t_start = time.perf_counter()
    t_first_token: float | None = None
    token_count = 0
    got_answer = False
    error: str | None = None

    try:
        async with session.post(
            f"{base}/api/chat/lightrag",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {
                    "ok": False,
                    "status": resp.status,
                    "error": text[:200],
                    "ttfb": 0,
                    "total": time.perf_counter() - t_start,
                    "tokens": 0,
                }

            async for line in resp.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                raw = decoded[5:].strip()
                if not raw:
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = evt.get("type")
                if etype == "token":
                    token_count += 1
                    if t_first_token is None:
                        t_first_token = time.perf_counter() - t_start
                elif etype == "answer":
                    got_answer = True
                elif etype == "error":
                    error = evt.get("content", "unknown")
                elif etype == "done":
                    break

    except Exception as exc:
        error = str(exc)

    total = time.perf_counter() - t_start
    return {
        "ok": got_answer and error is None,
        "status": 200,
        "error": error,
        "ttfb": t_first_token or total,
        "total": total,
        "tokens": token_count,
    }


# ── 压测主逻辑 ────────────────────────────────────────────────
async def run_level(
    base: str,
    tokens: list[str],
    concurrency: int,
    requests: int,
    course_id: str,
    message: str,
    history_turns: int,
) -> dict:
    connector = aiohttp.TCPConnector(limit=max(concurrency + 10, 20))
    sem = asyncio.Semaphore(max(1, concurrency))
    rng = random.Random()

    def _build_payload() -> dict:
        chosen_msg = rng.choice(DEFAULT_MESSAGES) if rng.random() < 0.7 else message
        return {
            "course_id": course_id,
            "message": chosen_msg,
            "history": build_history(history_turns),
            "chat_mode": "chat",
        }

    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _one_request() -> dict:
            async with sem:
                token = rng.choice(tokens)
                return await send_chat_sse(session, base, token, _build_payload())

        tasks = [_one_request() for _ in range(requests)]
        results = await asyncio.gather(*tasks)

    ok_results = [r for r in results if r["ok"]]
    err_results = [r for r in results if not r["ok"]]
    totals = [r["total"] for r in ok_results] or [0.0]
    ttfbs = [r["ttfb"] for r in ok_results] or [0.0]
    tokens_list = [r["tokens"] for r in ok_results] or [0]

    totals_sorted = sorted(totals)
    ttfbs_sorted = sorted(ttfbs)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    wall_clock = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "requests": requests,
        "ok": len(ok_results),
        "errors": len(err_results),
        "error_rate": f"{len(err_results) / requests * 100:.1f}%",
        "ttfb_p50": percentile(ttfbs_sorted, 0.5),
        "ttfb_p95": percentile(ttfbs_sorted, 0.95),
        "total_p50": percentile(totals_sorted, 0.5),
        "total_p95": percentile(totals_sorted, 0.95),
        "total_max": max(totals),
        "avg_tokens": statistics.mean(tokens_list),
        "qps": len(ok_results) / wall_clock if wall_clock > 0 else 0.0,
        "err_samples": [r["error"] for r in err_results[:3]],
    }


def print_report(rows: list[dict]):
    print()
    print("=" * 100)
    print(f"{'conc':>6} | {'reqs':>5} | {'ok':>4} | {'err%':>6} | "
          f"{'TTFB p50':>9} | {'TTFB p95':>9} | "
          f"{'Total p50':>10} | {'Total p95':>10} | {'Max':>8} | "
          f"{'QPS':>6} | {'avg_tok':>8}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['concurrency']:>6} | "
            f"{r['requests']:>5} | "
            f"{r['ok']:>4} | "
            f"{r['error_rate']:>6} | "
            f"{r['ttfb_p50']:>8.2f}s | "
            f"{r['ttfb_p95']:>8.2f}s | "
            f"{r['total_p50']:>9.2f}s | "
            f"{r['total_p95']:>9.2f}s | "
            f"{r['total_max']:>7.2f}s | "
            f"{r['qps']:>6.2f} | "
            f"{r['avg_tokens']:>8.1f}"
        )
    print("=" * 100)

    if any(r["err_samples"] for r in rows):
        print("\nerror samples:")
        for r in rows:
            for e in r["err_samples"]:
                print(f"  [conc={r['concurrency']}] {e}")


async def main():
    parser = argparse.ArgumentParser(description="课程 Agent 并发压测")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--per-level", type=int, default=DEFAULT_PER_LEVEL,
                        help="每个并发档位总共发起多少次请求（通过并发上限排队发送）")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS,
                        help="预创建用户数，用于模拟多用户并发")
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS,
                        help="每次请求附带历史轮次（每轮含 user+assistant）")
    parser.add_argument("--user-create-concurrency", type=int, default=20,
                        help="预创建用户时的并发度")
    parser.add_argument("--course", default=DEFAULT_COURSE)
    parser.add_argument("--message", default=DEFAULT_MSG)
    args = parser.parse_args()

    print(f"target: {args.base_url}")
    print(f"concurrency levels: {args.concurrency}")
    print(f"requests per level: {args.per_level}")
    print(f"user pool size: {args.users}")
    print(f"history turns: {args.history_turns}")
    print(f"course: {args.course}")
    print(f"message: {args.message[:60]}...")
    print()

    print("registering user pool...")
    tokens = await register_user_pool(
        args.base_url,
        max(1, args.users),
        parallel=max(1, args.user_create_concurrency),
    )
    if not tokens:
        raise RuntimeError("无法创建可用用户，请检查 /api/auth/register 与服务状态")
    print(f"user pool ready: {len(tokens)} users\n")

    rows = []
    for c in args.concurrency:
        actual_reqs = max(c, args.per_level)
        print(f">> conc={c}, sending {actual_reqs} requests...", end=" ", flush=True)
        row = await run_level(
            args.base_url,
            tokens,
            c,
            actual_reqs,
            args.course,
            args.message,
            max(0, args.history_turns),
        )
        print(f"done (ok={row['ok']}, err={row['errors']}, "
              f"p50={row['total_p50']:.2f}s, qps={row['qps']:.2f})")
        rows.append(row)

        await asyncio.sleep(1)

    print_report(rows)


if __name__ == "__main__":
    asyncio.run(main())
