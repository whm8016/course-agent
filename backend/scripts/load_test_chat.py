"""
Load test: simulate N students hitting POST /api/chat/lightrag concurrently.

Usage:
    python scripts/load_test_chat.py --concurrency 20 --join-code ABCD1234

If --join-code is omitted, the script assumes students are already enrolled.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp

BASE_URL = "http://localhost:8002/api"
PASSWORD = "loadtest123"

QUESTIONS = [
    "给出一道伏安特性的题目",
    "请解释欧姆定律的基本原理",
    "什么是电阻的串并联？举例说明",
    "基尔霍夫电压定律怎么用？",
    "电容器的充放电过程是怎样的？",
    "什么是交流电的有效值？",
    "请解释电磁感应定律",
    "变压器的工作原理是什么？",
    "RLC串联电路的谐振条件是什么？",
    "三相电路中线电压和相电压的关系？",
    "功率因数是什么？怎么提高？",
    "什么是戴维南定理？",
    "诺顿定理和戴维南定理的关系？",
    "叠加定理的使用条件是什么？",
    "什么是电感的自感和互感？",
    "电路中短路和断路的区别？",
    "什么是理想电压源和电流源？",
    "如何用网孔分析法求解电路？",
    "节点电压法的步骤是什么？",
    "请出一道关于功率计算的题目",
]


@dataclass
class Result:
    student_id: int
    success: bool
    ttft_ms: float = 0.0  # time to first token
    total_ms: float = 0.0
    answer_chars: int = 0
    error: str = ""
    got_busy: bool = False  # saw "服务繁忙"


async def register_or_login(session: aiohttp.ClientSession, idx: int) -> str | None:
    """Register student; if exists, login. Returns JWT token or None."""
    username = f"loadtest_student_{idx:02d}"
    display_name = f"压测学生{idx:02d}"

    # Try register
    try:
        async with session.post(f"{BASE_URL}/auth/register", json={
            "username": username,
            "password": PASSWORD,
            "display_name": display_name,
        }) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["token"]
            if resp.status == 409:
                pass  # already exists, login below
            elif resp.status == 429:
                await asyncio.sleep(2)
            else:
                text = await resp.text()
                print(f"  [student {idx:02d}] register failed: {resp.status} {text[:100]}")
    except Exception as e:
        print(f"  [student {idx:02d}] register error: {e}")

    # Login
    try:
        async with session.post(f"{BASE_URL}/auth/login", json={
            "username": username,
            "password": PASSWORD,
        }) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["token"]
            text = await resp.text()
            print(f"  [student {idx:02d}] login failed: {resp.status} {text[:100]}")
    except Exception as e:
        print(f"  [student {idx:02d}] login error: {e}")

    return None


async def join_course(session: aiohttp.ClientSession, token: str, join_code: str, idx: int) -> bool:
    """Join the course. Returns True if enrolled."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with session.post(f"{BASE_URL}/courses/join", json={
            "join_code": join_code,
        }, headers=headers) as resp:
            if resp.status == 200:
                return True
            text = await resp.text()
            print(f"  [student {idx:02d}] join failed: {resp.status} {text[:100]}")
            return False
    except Exception as e:
        print(f"  [student {idx:02d}] join error: {e}")
        return False


async def simulate_student(
    session: aiohttp.ClientSession,
    token: str,
    question: str,
    idx: int,
) -> Result:
    """Send one chat request and collect metrics."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "course_id": "mycourse",
        "message": question,
        "history": [],
        "chat_mode": "chat",
        "tools": ["rag"],
    }

    result = Result(student_id=idx, success=False)
    t0 = time.perf_counter()

    try:
        async with session.post(
            f"{BASE_URL}/chat/lightrag",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                result.error = f"HTTP {resp.status}: {text[:200]}"
                result.total_ms = (time.perf_counter() - t0) * 1000
                return result

            first_token = False
            answer_parts = []
            async for line in resp.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data: "):
                    continue
                raw = decoded[6:]
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "token" and not first_token:
                    first_token = True
                    result.ttft_ms = (time.perf_counter() - t0) * 1000

                if etype == "token":
                    answer_parts.append(event.get("content", ""))

                if etype == "answer":
                    content = event.get("content", "")
                    if "服务暂时不可用" in content or "服务繁忙" in content:
                        result.got_busy = True
                        result.error = "AI 服务繁忙"

                if etype == "error":
                    result.error = event.get("content", "unknown error")

                if etype == "done":
                    result.success = not result.got_busy and not result.error
                    break

            result.answer_chars = sum(len(p) for p in answer_parts)
            if not result.error and result.answer_chars == 0 and not result.success:
                result.error = "no answer received"

    except asyncio.TimeoutError:
        result.error = "timeout (180s)"
    except Exception as e:
        result.error = str(e)[:200]

    result.total_ms = (time.perf_counter() - t0) * 1000
    return result


async def run_batch(
    session: aiohttp.ClientSession,
    tokens: list[str],
    concurrency: int,
    run_id: str = "",
) -> list[Result]:
    """Run one batch of concurrent requests."""
    tasks = []
    for i in range(concurrency):
        token = tokens[i % len(tokens)]
        base_q = QUESTIONS[i % len(QUESTIONS)]
        # Append run_id to bypass FAQ cache
        question = f"{base_q}（测试批次{run_id}_{i}）" if run_id else base_q
        tasks.append(simulate_student(session, token, question, i + 1))
    return await asyncio.gather(*tasks)


def print_report(results: list[Result], concurrency: int):
    """Print a summary table."""
    total = len(results)
    success = sum(1 for r in results if r.success)
    busy = sum(1 for r in results if r.got_busy)
    errors = sum(1 for r in results if r.error and not r.got_busy)
    ttfts = [r.ttft_ms for r in results if r.ttft_ms > 0]
    totals = [r.total_ms for r in results if r.total_ms > 0]

    print(f"\n{'='*60}")
    print(f"  并发数: {concurrency}")
    print(f"  成功: {success}/{total}  |  服务繁忙: {busy}  |  其他错误: {errors}")
    if ttfts:
        print(f"  首token延迟: min={min(ttfts):.0f}ms  avg={sum(ttfts)/len(ttfts):.0f}ms  max={max(ttfts):.0f}ms")
    if totals:
        print(f"  总耗时:     min={min(totals):.0f}ms  avg={sum(totals)/len(totals):.0f}ms  max={max(totals):.0f}ms")
    print(f"{'='*60}")

    if busy or errors:
        print("  失败详情:")
        for r in results:
            if r.error:
                print(f"    student {r.student_id:02d}: {r.error}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Load test /api/chat/lightrag")
    parser.add_argument("--concurrency", "-c", type=int, default=20)
    parser.add_argument("--join-code", "-j", type=str, default="")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--rounds", "-r", type=int, default=1,
                        help="Run multiple rounds with increasing concurrency (5,10,15,20)")
    args = parser.parse_args()

    concurrency = args.concurrency
    if args.rounds > 1:
        levels = [concurrency] * args.rounds
    else:
        levels = [concurrency]

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: Register/login students
        max_students = max(levels)
        print(f"[Phase 1] 注册/登录 {max_students} 个测试学生...")
        tokens: list[str] = []
        for i in range(1, max_students + 1):
            token = await register_or_login(session, i)
            if token:
                tokens.append(token)
            else:
                print(f"  WARNING: student {i:02d} auth failed, skipping")
            # register rate limit = 10/min, login = 15/min; be conservative
            if i % 8 == 0:
                print(f"  (等待 65s 避免注册限流...)")
                await asyncio.sleep(65)
            else:
                await asyncio.sleep(0.3)

        print(f"  获得 {len(tokens)} 个有效 token")

        if not tokens:
            print("ERROR: 没有可用的学生账号，退出")
            return

        # Phase 2: Join course (if join_code provided)
        if args.join_code:
            print(f"\n[Phase 2] 加入课程 (join_code={args.join_code})...")
            for i, token in enumerate(tokens):
                await join_course(session, token, args.join_code, i + 1)
                if (i + 1) % 5 == 0:
                    await asyncio.sleep(0.5)
            print("  完成")

        # Phase 3: Run load test
        for li, level in enumerate(levels):
            actual = min(level, len(tokens))
            run_id = f"{int(time.time())}_{li}"
            print(f"\n[Phase 3] 压测开始: {actual} 并发请求到 /api/chat/lightrag ...")
            t0 = time.perf_counter()
            results = await run_batch(session, tokens[:actual], actual, run_id=run_id)
            wall_time = (time.perf_counter() - t0) * 1000
            print(f"  批次完成，总墙钟时间: {wall_time:.0f}ms")
            print_report(results, actual)

            # Short gap between rounds (simulate continuous classroom usage)
            if li < len(levels) - 1:
                print("  等待 5s 后开始下一轮...")
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
