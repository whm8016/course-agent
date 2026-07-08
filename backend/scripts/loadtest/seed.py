"""压测数据 seed：造 1 教师(admin) + M 课程 + N 学生（各入全部课程），缓存 tokens.json。

照搬 tests/conftest.py 的 API 序列（注册→DBA 升 admin→建课→join_code→入课），但：
  - 教师升 admin 走 asyncpg 直连 postgres UPDATE（生产无 admin 提权端点，H-18 后只能
    DBA 通道——core/db/auth.py:76 注释明确）；
  - 不 import backend 代码，纯外部客户端（httpx + asyncpg），零副作用、可在宿主直接跑。

预缓存 token：Locust on_start 直接读 tokens.json，**不在压测期登录**——/api/auth/login
走 bcrypt 同步校验，CPU-bound 会挡住 gevent greenlet，并发登录会把 Locust worker 卡死。

限流处理：auth 的 register/login 有 per-IP 限流（@limiter.limit）。seed 快速批量注册会
撞 429，故 register/login 均带 429 退避重试，学生循环间隔节流。

用法（宿主 Windows，backend/venv 已含 httpx/asyncpg）：
  ./backend/venv/Scripts/python.exe backend/scripts/loadtest/seed.py \\
      --base-url http://localhost:8000 \\
      --pg-url "postgresql://postgres:postgres@localhost:5433/course_agent_loadtest" \\
      --students 20 --courses 3 --out tokens.json

前置：docker compose -f docker-compose.loadtest.yml up（backend + postgres 已 healthy、
表已由 backend lifespan init_db 建好）。
幂等：重跑前建议 `docker compose ... down -v` 清库；脚本对「用户/课程已存在」做了
fallback（register 失败→login、建课已存在→跳过），但最干净仍是清库后重 seed。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import asyncpg
import httpx

DEFAULT_PASSWORD = "testpass123"
# 限流退避：auth register/login 有 per-IP 限流，批量注册会 429，重试到窗口放行。
_RATE_RETRY = 12
_RATE_BACKOFF = 2.0


async def _post(client: httpx.AsyncClient, path: str, payload: dict,
                headers: dict | None = None):
    """带 429 退避重试的 POST。非限流错误立即返回；429 线性退避重试。"""
    r = None
    for attempt in range(_RATE_RETRY):
        r = await client.post(path, json=payload, headers=headers)
        if r.status_code != 429:
            return r
        await asyncio.sleep(_RATE_BACKOFF * (attempt + 1))  # 线性退避
    return r  # 重试耗尽（仍 429）


async def register(client: httpx.AsyncClient, username: str,
                   password: str = DEFAULT_PASSWORD, display: str = "T") -> str | None:
    """注册，返回 token；用户名已存在返回 None（由调用方走 login）；429 限流退避重试。"""
    r = await _post(client, "/api/auth/register",
                    {"username": username, "password": password, "display_name": display})
    if r.status_code == 200:
        return r.json()["token"]
    return None  # 非 200/429（已存在/冲突）或 429 重试耗尽 → 调用方 fallback login


async def login(client: httpx.AsyncClient, username: str,
                password: str = DEFAULT_PASSWORD) -> str:
    r = await _post(client, "/api/auth/login",
                    {"username": username, "password": password})
    if r.status_code != 200:
        raise RuntimeError(f"login {username} failed: {r.status_code} {r.text[:120]}")
    return r.json()["token"]


async def ensure_user(client: httpx.AsyncClient, username: str, display: str) -> str:
    """注册或复用：register 成功拿 token；已存在则 login。返回注册/登录 token。"""
    token = await register(client, username, display=display)
    if token is None:
        token = await login(client, username)
    return token


# ---- DBA 通道：升 admin（无生产端点，直连 postgres UPDATE）----
async def promote_admin(pg_url: str, username: str) -> None:
    conn = await asyncpg.connect(pg_url)
    try:
        res = await conn.execute(
            "UPDATE users SET role='admin', is_admin=TRUE WHERE username=$1",
            username,
        )
        if res == "UPDATE 0":
            raise RuntimeError(f"promote {username}: user not found in DB (先 register?)")
    finally:
        await conn.close()


async def main(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        # 1. 教师：register/login → DBA 升 admin → 重新 login 拿带 admin 权限的 token
        teacher_user = "teacher_loadtest"
        await ensure_user(client, teacher_user, display="压测教师")
        await promote_admin(args.pg_url, teacher_user)
        await asyncio.sleep(_RATE_BACKOFF)  # 避免紧跟的 login 撞限流
        teacher_token = await login(client, teacher_user)
        headers_teacher = {"Authorization": f"Bearer {teacher_token}"}

        # 2. M 个课程（H-10 复验要 >2 个 course_id 才能触发实例池 evict）
        courses: list[dict] = []
        for i in range(args.courses):
            cid = f"lt_course_{i:02d}"
            r = await client.post(
                "/api/admin/kb",
                headers=headers_teacher,
                json={"course_id": cid, "name": f"压测课程{i}", "is_visible": True},
            )
            if r.status_code not in (200, 201):
                # 已存在（重跑）→ 容错继续；其他错误暴露
                if r.status_code not in (400, 409):
                    raise RuntimeError(f"create kb {cid} failed: {r.status_code} {r.text[:120]}")
            r2 = await client.post(
                f"/api/teacher/courses/{cid}/join-code",
                headers=headers_teacher,
            )
            if r2.status_code != 200:
                raise RuntimeError(f"join-code {cid} failed: {r2.status_code} {r2.text[:120]}")
            courses.append({
                "course_id": cid,
                "join_code": r2.json()["join_code"],
                "name": f"压测课程{i}",
            })

        # 3. N 学生，各入全部课程（用各自的注册 token）
        students: list[dict] = []
        for i in range(args.students):
            uname = f"stu_{i:03d}"
            token = await ensure_user(client, uname, display=f"压测学生{i}")
            await asyncio.sleep(args.throttle)  # 节流：避免快速连发撞 per-IP 注册限流
            headers = {"Authorization": f"Bearer {token}"}
            enrolled: list[str] = []
            for cr in courses:
                rj = await _post(client, "/api/courses/join",
                                 {"join_code": cr["join_code"]}, headers=headers)
                if rj.status_code == 200:
                    enrolled.append(cr["course_id"])
                # 已入课（重跑）→ 忽略非致命
            students.append({"username": uname, "token": token, "course_ids": enrolled})
            if (i + 1) % 5 == 0:
                print(f"  ...已造 {i + 1}/{args.students} 学生")

    data = {
        "base_url": args.base_url,
        "teacher": {
            "username": teacher_user,
            "token": teacher_token,
            "course_ids": [c["course_id"] for c in courses],
        },
        "courses": courses,
        "students": students,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(
        f"seeded {len(students)} students + {len(courses)} courses "
        f"(teacher={teacher_user}) → {args.out}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="压测数据 seed（教师+课程+学生，缓存 tokens.json）")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="nginx 入口（默认 http://localhost:8000）")
    p.add_argument("--pg-url",
                   default="postgresql://postgres:postgres@localhost:5433/course_agent_loadtest",
                   help="postgres 连接串（asyncpg 用 postgresql://，勿带 +asyncpg）")
    p.add_argument("--students", type=int, default=20, help="学生数（默认 20）")
    p.add_argument("--courses", type=int, default=3, help="课程数（默认 3，H-10 复验要 >2）")
    p.add_argument("--throttle", type=float, default=0.5,
                   help="每个学生注册后节流秒数（避免撞 per-IP 限流，默认 0.5）")
    p.add_argument("--out", default="tokens.json", help="输出文件（默认 tokens.json）")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
