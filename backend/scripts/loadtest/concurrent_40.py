"""上线验收：20 学生并发 × 每人 2 问 = 恰好 40 次提问（确定性，非 Locust 持续压测）。

与 locustfile.py 的分工：Locust 是"随机持续负载"（稳态 RPS / 分位数 / 限流行为），
本脚本是"确定性验收"——精确控制 20 个学生**同一时刻**开问、每人串行 2 问（像真实
学生：问完一题等答案再问下一题）、总计恰好 40 次、每问一个不同问题，输出 PASS/FAIL
供上线 go/no-go。不经过 Locust 的随机 task 循环，学生与 token 1:1（不 random.choice
撞车），样本可复现。

判定逻辑复用 locustfile 的 H-15 语义：
  token/answer 事件 → 有效内容；done → 正常收尾；
  error 事件 → 合法响应（fire success，但单独计数并列明细——上线验收中值得人工看）；
  连接断开且无任何事件 = silence → FAIL（真正的静默 bug，H-15 要 0）。
  另单列 no_done 计数：有内容但既无 done 也无 error（干净 EOF 截断，H-15 口径算
  PASS——与 locustfile 可比——但可能是 is_disconnected 误报截断，验收时需人工看）。

请求体对齐 locustfile.chat_sse：chat_mode=chat + tools=[]（纯压 chat 流式链路，
不查 KB/不调工具）；schema 见 api/chat.py:33 ChatRequest（session_id 可选不传，
服务端无会话时回落 history=[] 正常走）。

用法（宿主 Windows，backend/.venv 已含 httpx）：
  ./backend/.venv/Scripts/python.exe backend/scripts/loadtest/concurrent_40.py
前置：loadtest compose 已 healthy + seed.py 已产出 tokens.json（默认同目录）。
退出码：全过=0；任一 FAIL（非200/silence/异常）=1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import httpx

# 40 个互不重复的中文提问（下标 = 学生序号*2 + 第几问）。tools=[] 不触发 KB/工具，
# 问题走通用计算机基础知识即可，不依赖压测课程内容。
QUESTIONS: list[str] = [
    "请用一句话解释快速排序的核心思想",
    "哈希表最坏情况为什么会退化成 O(n)？",
    "TCP 三次握手为什么不是两次？",
    "进程和线程的区别是什么？",
    "数据库索引为什么会加快查询？",
    "什么是死锁？举一个生活中的例子",
    "二分查找的前提条件是什么？",
    "HTTP 和 HTTPS 的区别在哪里？",
    "栈和队列分别适合什么场景？",
    "什么是递归？它有什么风险？",
    "为什么很多语言的数组下标从 0 开始？",
    "虚拟内存解决了什么问题？",
    "事务的 ACID 分别指什么？",
    "广度优先搜索和深度优先搜索有什么不同？",
    "Python 的列表和元组有什么区别？",
    "什么是时间复杂度的大 O 记号？",
    "SQL 中 WHERE 和 HAVING 的区别是什么？",
    "图的拓扑排序用在什么场景？",
    "面向对象的封装有什么好处？",
    "DNS 解析的过程是怎样的？",
    "动态规划和贪心算法的核心区别是什么？",
    "为什么学习编程要先学数据结构？",
    "链表相比数组的优缺点是什么？",
    "消息队列的作用是什么？",
    "Redis 为什么快？",
    "什么是 API？举一个例子",
    "云计算的 IaaS 和 PaaS 有什么区别？",
    "编译器和解释器的区别是什么？",
    "什么是版本控制？为什么需要 Git？",
    "单元测试的意义是什么？",
    "归并排序的时间复杂度为什么是 O(n log n)？",
    "什么是负载均衡？",
    "Cookie 和 Session 的区别是什么？",
    "为什么数据库要分库分表？",
    "什么是 RESTful API？",
    "二叉搜索树退化成链表会怎样？",
    "缓存穿透是什么意思？怎么防？",
    "乐观锁和悲观锁的区别是什么？",
    "什么是微服务架构？",
    "初学编程应该先学哪门语言？为什么？",
]


@dataclass
class AskResult:
    student: str          # stu_000 形式用户名
    q_idx: int            # 全局问题下标（0..39）
    status: str           # "ok" / "http_<code>" / "silent" / "exception"
    http_status: int = 0
    ttft_ms: float | None = None   # 首 token/answer 延迟；无内容则 None
    total_ms: float = 0.0
    got_error: bool = False        # 收到 error 事件（合法响应，单独计数供人工看）
    no_done: bool = False          # 有内容但无 done/error 收尾（截断嫌疑，H-15 口径仍 PASS）
    error_detail: str = ""
    note: str = ""                 # 问题摘要

    @property
    def pass_(self) -> bool:
        return self.status == "ok"


async def ask_once(
    client: httpx.AsyncClient,
    base: str,
    headers: dict,
    course_id: str,
    question: str,
    student: str,
    q_idx: int,
    timeout_s: float,
) -> AskResult:
    """单次 SSE 提问。事件判定语义与 locustfile._finish 一致（H-15）。
    超时两层：client 级 httpx.Timeout（相邻 SSE 读间隔，构造见 main）+ asyncio.timeout
    单问墙钟——read timeout 只限相邻两次读之间，服务端"每隔 N 秒滴一个 token"的挂死
    不会触发它，没有墙钟兜底整个验收会无限挂起。不传 per-request timeout：标量会
    整体覆盖 client 级 Timeout 让 connect=10 失效（httpx build_request 实测行为）。"""
    body = {
        "course_id": course_id,
        "message": question,
        "chat_mode": "chat",
        "history": [],
        "tools": [],
    }
    t0 = time.perf_counter()
    res = AskResult(student=student, q_idx=q_idx, status="exception",
                    note=question[:18])
    try:
        async with asyncio.timeout(timeout_s), \
                client.stream("POST", f"{base}/api/chat",
                              headers=headers, json=body) as r:
            res.http_status = r.status_code
            if r.status_code != 200:
                snippet = (await r.aread())[:120].decode("utf-8", "replace")
                res.status = f"http_{r.status_code}"
                res.error_detail = snippet
                res.total_ms = (time.perf_counter() - t0) * 1000
                return res
            got_content = got_error = got_done = False
            async for raw in r.aiter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    evt = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t in ("token", "answer"):
                    got_content = True
                    if res.ttft_ms is None:
                        res.ttft_ms = (time.perf_counter() - t0) * 1000
                elif t == "error":
                    got_error = True
                    res.error_detail = str(evt.get("message") or evt)[:160]
                elif t == "done":
                    got_done = True
                    break
            res.total_ms = (time.perf_counter() - t0) * 1000
            if got_content or got_done or got_error:
                res.status = "ok"
                res.got_error = got_error
                res.no_done = got_content and not (got_done or got_error)
            else:
                res.status = "silent"   # 连接走完但无任何有效事件（H-15 真失败）
    except TimeoutError:   # asyncio.timeout 墙钟（3.11+ 即内建 TimeoutError；httpx.ReadTimeout 自成体系不被此捕获）
        res.total_ms = (time.perf_counter() - t0) * 1000
        res.error_detail = f"wall-clock >{timeout_s:.0f}s"
    except Exception as exc:
        res.total_ms = (time.perf_counter() - t0) * 1000
        res.error_detail = f"{type(exc).__name__}: {exc}"[:160]
    return res


async def student_runner(
    client: httpx.AsyncClient,
    base: str,
    stu: dict,
    idx: int,
    per_student: int,
    timeout_s: float,
) -> list[AskResult]:
    """一个学生：串行问 per_student 题（问完等答案再问下一题），课程固定取其
    enrolled 列表按 idx 轮转的一个（确定性，可复现）。"""
    headers = {"Authorization": f"Bearer {stu['token']}"}
    course_ids = stu.get("course_ids") or [""]
    course_id = course_ids[idx % len(course_ids)]
    out: list[AskResult] = []
    for j in range(per_student):
        q_idx = idx * per_student + j
        question = QUESTIONS[q_idx]
        r = await ask_once(client, base, headers, course_id, question,
                           stu["username"], q_idx, timeout_s)
        mark = "PASS" if r.pass_ else "FAIL"
        flags = []
        if r.got_error:
            flags.append("error_event")
        if r.no_done:
            flags.append("no_done")
        if not r.pass_:
            flags.append(r.error_detail[:80])
        err = f" [{'|'.join(flags)}]" if flags else ""
        ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms is not None else "-"
        print(f"[{mark}] {r.student} Q{q_idx:02d} {r.status:10s} "
              f"ttft={ttft:>8s} total={r.total_ms:7.0f}ms{err}")
        out.append(r)
    return out


def pct(sorted_vals: list[float], p: float) -> float:
    """nearest-rank 分位数（40 样本够用，不引依赖）。"""
    if not sorted_vals:
        return float("nan")
    k = max(1, math.ceil(p * len(sorted_vals)))
    return sorted_vals[k - 1]


async def main(args: argparse.Namespace) -> int:
    total_n = args.students * args.per_student
    if args.students < 1 or args.per_student < 1:
        print("students/per-student 必须 ≥1")
        return 2
    if total_n > len(QUESTIONS):
        # QUESTIONS 按 q_idx 直接索引，越界会穿透 gather 打飞全部在飞结果
        print(f"students×per-student={total_n} 超过题库 {len(QUESTIONS)} 题，"
              f"请扩充 QUESTIONS 或调小参数")
        return 2
    with open(args.tokens, encoding="utf-8") as f:
        data = json.load(f)
    students = data["students"]
    if len(students) < args.students:
        print(f"tokens.json 只有 {len(students)} 学生（要 {args.students}），先跑 seed.py")
        return 2
    students = students[:args.students]
    base = data["base_url"].rstrip("/")
    print(f"==== 上线验收：{args.students} 学生并发 × {args.per_student} 问 "
          f"= {total_n} 次（base={base}）====")
    t0 = time.perf_counter()
    # 每学生一个 task 同时启动（"同时提问"）；连接池上限放到学生数，避免复用排队
    limits = httpx.Limits(max_connections=args.students * 2,
                          max_keepalive_connections=args.students)
    timeout = httpx.Timeout(args.timeout, connect=10)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        grouped = await asyncio.gather(*[
            student_runner(client, base, stu, i, args.per_student, args.timeout)
            for i, stu in enumerate(students)
        ])
    flat = [r for rs in grouped for r in rs]
    wall = time.perf_counter() - t0

    passed = sum(r.pass_ for r in flat)
    silent = sum(r.status == "silent" for r in flat)
    bad_http = sum(r.status.startswith("http_") for r in flat)
    exceptions = sum(r.status == "exception" for r in flat)
    err_events = sum(r.got_error for r in flat)
    no_done = sum(r.no_done for r in flat)
    ttfts = sorted(r.ttft_ms for r in flat if r.ttft_ms is not None)
    # 延迟分布只看成功样本：失败请求混入会使分位数失真（30s 排队 500 拉高、秒回 429 拉低）
    totals = sorted(r.total_ms for r in flat if r.status == "ok")

    print(f"\n---- 汇总（墙钟 {wall:.1f}s）----")
    print(f"{passed}/{total_n} PASS   FAIL: 非200={bad_http} silence={silent} "
          f"异常={exceptions}   error事件={err_events} 无done截断={no_done}")
    if ttfts:
        print(f"TTFT  P50={pct(ttfts, .50):8.0f}ms  P95={pct(ttfts, .95):8.0f}ms "
              f"max={ttfts[-1]:.0f}ms")
    if totals:   # 全失败跑法（排障场景）下无成功样本
        print(f"total P50={pct(totals, .50):8.0f}ms  P95={pct(totals, .95):8.0f}ms "
              f"max={totals[-1]:.0f}ms")
    if err_events:
        print("error 事件明细（合法响应，需人工确认成因）：")
        for r in flat:
            if r.got_error:
                print(f"  {r.student} Q{r.q_idx:02d}: {r.error_detail}")
    verdict = "PASS" if passed == total_n else "FAIL"
    warns = []
    if err_events:
        warns.append("含 error 事件，成因见上")
    if no_done:
        warns.append(f"{no_done} 问有内容无 done（截断嫌疑）")
    warn = f"（{'；'.join(warns)}）" if warns else ""
    print(f"\n==== 上线验收判定：{verdict} {warn} ====")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict, "total": total_n, "passed": passed,
            "silent": silent, "bad_http": bad_http, "exceptions": exceptions,
            "error_events": err_events, "no_done": no_done, "wall_sec": round(wall, 1),
            "ttft_p50_ms": round(pct(ttfts, .50)) if ttfts else None,
            "ttft_p95_ms": round(pct(ttfts, .95)) if ttfts else None,
            "total_p50_ms": round(pct(totals, .50)) if totals else None,
            "total_p95_ms": round(pct(totals, .95)) if totals else None,
            "results": [{
                "student": r.student, "q": r.q_idx, "status": r.status,
                "http": r.http_status, "got_error": r.got_error,
                "no_done": r.no_done,
                "ttft_ms": round(r.ttft_ms) if r.ttft_ms is not None else None,
                "total_ms": round(r.total_ms), "error": r.error_detail,
            } for r in flat],
        }, f, ensure_ascii=False, indent=2)
    print(f"明细 JSON → {os.path.abspath(args.out)}")
    return 0 if passed == total_n else 1


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(__file__)
    p = argparse.ArgumentParser(description="20 学生 × 2 问 = 40 次并发上线验收")
    p.add_argument("--tokens", default=os.path.join(here, "tokens.json"))
    p.add_argument("--students", type=int, default=20)
    p.add_argument("--per-student", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0,
                   help="单问墙钟总时长上限秒（ask_once 内 asyncio.timeout 兜底；client 级"
                        " read timeout 同值，防服务端'滴 token 式挂死'永不触发读超时）")
    p.add_argument("--out", default=os.path.join(here, "concurrent_40_result.json"))
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
