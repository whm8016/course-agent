"""模拟学生连续提问，助手回复走真实模型，展示 mem0 实际存了什么。

学生侧问题可预设；每轮助手回答由 ChatPipeline → run_agent_loop 真实生成（与线上一致），
不是脚本写死的假回复。生成后再喂 mem0，打印抽取结果。

Usage（必须在 Docker backend 容器内跑，宿主机 conda 缺 sqlalchemy/mem0ai 等依赖）:

    # 项目根目录
    docker compose exec backend .venv/bin/python -m scripts.demo_mem0_student_qa --cleanup-before --cleanup-after
    docker compose exec backend .venv/bin/python -m scripts.demo_mem0_student_qa --turns 3 --course-id mycourse
    docker compose exec backend .venv/bin/python -m scripts.demo_mem0_student_qa --mode consolidate --course-id mycourse --no-rag
    docker compose exec backend .venv/bin/python -m scripts.demo_mem0_student_qa --out results/mem0_demo.json

    若未启动栈：先在项目根 docker compose up -d

模式:
  direct（默认）: 每轮模型答完后立即 mem0.add，立刻看到抽取结果。
  consolidate:    每轮 record_episode，全部问完后 consolidate 批量升格到 mem0（线上路径）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# 仅预设「学生提问」；助手回答由模型实时生成
STUDENT_QUESTIONS: list[str] = [
    "老师好，我是大二学生小王。第一次做实验三，两个 100Ω 电阻并联，"
    "我算成 200Ω 了，是不是并联和串联一样直接相加？",
    "明白了。那戴维南等效里，开路电压 Uoc 和等效电阻 Req 我总是搞混，"
    "能简单说一下步骤吗？",
    "叠加定理是不是所有电路都能用？我们实验报告要求写适用条件。",
    "好的，谢谢。",
    "实验五前我又把并联电阻算成相加了，R1=200Ω、R2=200Ω 我写成 400Ω，"
    "这次是不是又犯了同样的错误？",
    "我打算周末再练一遍实验三和实验五的电路，有没有复习顺序建议？",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _normalize_memory_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "id": item.get("id"),
        "memory": item.get("memory"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "user_id": item.get("user_id"),
        "course_id": meta.get("course_id") or item.get("course_id"),
        "event": item.get("event"),
        "hash": item.get("hash"),
    }


async def _fetch_all_memories(user_id: str, course_id: str) -> list[dict[str, Any]]:
    from core.memory.mem0_client import get_memory, normalize_results

    m = get_memory()
    filters: dict[str, str] = {"user_id": user_id}
    if course_id:
        filters["course_id"] = course_id
    resp = await m.get_all(filters=filters, top_k=200)
    items = normalize_results(resp)
    return [_normalize_memory_item(i) for i in items if isinstance(i, dict)]


async def _delete_all_memories(user_id: str) -> None:
    from core.memory.mem0_client import get_memory

    m = get_memory()
    await m.delete_all(user_id=user_id)
    logger.info("[demo] 已清空 user_id=%s 的全部 mem0 记忆", user_id)


async def _add_one_turn(
    user_id: str,
    course_id: str,
    user_msg: str,
    assistant_msg: str,
) -> dict[str, Any]:
    from core.memory.mem0_client import get_memory, normalize_results, should_skip_user_message

    if should_skip_user_message(user_msg):
        return {"skipped": True, "reason": "gate", "stored_this_turn": []}

    m = get_memory()
    result = await m.add(
        [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        user_id=user_id,
        metadata={"course_id": course_id},
    )
    stored = [_normalize_memory_item(i) for i in normalize_results(result) if isinstance(i, dict)]
    return {"skipped": False, "stored_this_turn": stored}


async def _ask_model(
    *,
    user_id: str,
    course_id: str,
    session_id: str,
    user_msg: str,
    history: list[dict[str, Any]],
    enabled_tools: list[str],
    rag_mode: str,
) -> dict[str, Any]:
    """走 orchestrator → ChatPipeline → run_agent_loop，返回模型真实回复。"""
    from core.agent.mode_normalize import normalize_mode
    from core.context import UnifiedContext
    from core.memory.mem0_client import build_memory_context
    from core.orchestrator import get_orchestrator
    from core.stream import StreamEventType

    memory_context = await build_memory_context(user_id, user_msg, course_id=course_id)
    ctx = UnifiedContext(
        course_id=course_id or "general",
        user_id=user_id,
        session_id=session_id,
        user_message=user_msg,
        conversation_history=history,
        mode=normalize_mode("chat"),
        enabled_tools=enabled_tools,
        rag_mode=rag_mode,
        memory_context=memory_context,
        metadata={"turn_id": f"demo-{uuid.uuid4().hex[:12]}"},
    )

    answer = ""
    token_parts: list[str] = []
    tools_used: list[str] = []
    error = ""

    async for event in get_orchestrator().handle(ctx):
        if event.type == StreamEventType.ANSWER:
            answer = str(event.payload.get("content", "") or "")
        elif event.type == StreamEventType.TOKEN:
            token_parts.append(str(event.payload.get("content", "") or ""))
        elif event.type == StreamEventType.TOOL_CALL:
            tool = str(event.payload.get("tool") or event.payload.get("name") or "")
            if tool:
                tools_used.append(tool)
        elif event.type == StreamEventType.ERROR:
            error = str(event.payload.get("content") or event.payload or "")

    final = (answer or "".join(token_parts)).strip()
    if error and not final:
        final = f"[模型错误] {error}"

    return {
        "assistant_msg": final,
        "tools_used": list(dict.fromkeys(tools_used)),
        "memory_context_len": len(memory_context),
        "error": error or None,
    }


def _print_block(title: str, payload: Any) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _print_turn_header(round_no: int, user_msg: str, assistant_msg: str) -> None:
    print("\n" + "-" * 72)
    print(f"【第 {round_no} 轮】")
    print("-" * 72)
    print(f"学生: {user_msg}")
    print(f"助手（模型生成）: {assistant_msg}")


async def _resolve_user_id(*, user_id: str | None, username: str | None) -> str:
    if user_id:
        return user_id
    if username:
        from sqlalchemy import select

        from core.db.database import AsyncSessionLocal, User

        async with AsyncSessionLocal() as db:
            row = (await db.execute(select(User).where(User.username == username))).scalars().first()
            if not row:
                raise SystemExit(f"用户不存在: username={username!r}")
            logger.info("使用数据库用户 id=%s username=%s", row.id, row.username)
            return row.id
    return f"demo-mem0-{uuid.uuid4().hex[:10]}"


def _append_history(history: list[dict[str, Any]], user_msg: str, assistant_msg: str) -> None:
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # 与 api/chat.py 对齐，最多保留最近 10 轮（20 条）
    if len(history) > 20:
        del history[:-20]


async def _run_direct(args: argparse.Namespace, user_id: str) -> dict[str, Any]:
    questions = STUDENT_QUESTIONS[: args.turns]
    session_id = args.session_id or f"demo-{uuid.uuid4().hex[:10]}"
    history: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []

    before = await _fetch_all_memories(user_id, args.course_id)
    _print_block(f"【开始前】mem0 已有记忆 user={user_id} course={args.course_id}", before)

    for i, user_msg in enumerate(questions, start=1):
        print(f"\n>>> 第 {i} 轮：调用模型生成回答...")
        model_out = await _ask_model(
            user_id=user_id,
            course_id=args.course_id,
            session_id=session_id,
            user_msg=user_msg,
            history=history,
            enabled_tools=args.enabled_tools,
            rag_mode=args.rag_mode,
        )
        assistant_msg = model_out["assistant_msg"]
        if not assistant_msg:
            raise SystemExit(f"第 {i} 轮模型未返回有效回答，请检查 LLM 配置。")

        _print_turn_header(i, user_msg, assistant_msg)
        if model_out.get("tools_used"):
            print(f"工具: {', '.join(model_out['tools_used'])}")

        add_result = await _add_one_turn(user_id, args.course_id, user_msg, assistant_msg)
        all_memories = await _fetch_all_memories(user_id, args.course_id)
        _append_history(history, user_msg, assistant_msg)

        round_snap = {
            "round": i,
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "model": model_out,
            "add_result": add_result,
            "memory_count_after": len(all_memories),
            "all_memories_after": all_memories,
        }
        rounds.append(round_snap)

        if add_result.get("skipped"):
            print("\n>>> 本轮被门控跳过（如「好的」「谢谢」），mem0 未写入。")
        else:
            stored = add_result.get("stored_this_turn") or []
            print(f"\n>>> 本轮 mem0 抽取/决策: {len(stored)} 条")
            for j, item in enumerate(stored, start=1):
                print(f"    [{j}] ({item.get('event') or '?'}) {item.get('memory')}")

        print(f"\n>>> 累计 mem0 记忆总数: {len(all_memories)}")
        for j, item in enumerate(all_memories, start=1):
            print(f"    #{j} {item.get('memory')}")

    search_ctx = ""
    if args.search_query:
        from core.memory.mem0_client import build_memory_context

        search_ctx = await build_memory_context(
            user_id, args.search_query, course_id=args.course_id, top_k=5
        )
        _print_block(f"【检索注入】query={args.search_query!r}", search_ctx or "(空)")

    after = await _fetch_all_memories(user_id, args.course_id)
    return {
        "mode": "direct",
        "user_id": user_id,
        "course_id": args.course_id,
        "session_id": session_id,
        "rounds": rounds,
        "before_count": len(before),
        "after_count": len(after),
        "final_memories": after,
        "search_context": search_ctx,
    }


async def _run_consolidate(args: argparse.Namespace, user_id: str) -> dict[str, Any]:
    from core.db.database import AsyncSessionLocal
    from core.memory.consolidation import consolidate
    from core.memory.episodic import record_episode

    questions = STUDENT_QUESTIONS[: args.turns]
    session_id = args.session_id or f"demo-{uuid.uuid4().hex[:10]}"
    history: list[dict[str, Any]] = []
    qa_pairs: list[tuple[str, str]] = []

    before = await _fetch_all_memories(user_id, args.course_id)
    _print_block(f"【开始前】mem0 已有记忆 user={user_id} course={args.course_id}", before)

    turn_records: list[dict[str, Any]] = []
    for i, user_msg in enumerate(questions, start=1):
        print(f"\n>>> 第 {i} 轮：调用模型生成回答...")
        model_out = await _ask_model(
            user_id=user_id,
            course_id=args.course_id,
            session_id=session_id,
            user_msg=user_msg,
            history=history,
            enabled_tools=args.enabled_tools,
            rag_mode=args.rag_mode,
        )
        assistant_msg = model_out["assistant_msg"]
        if not assistant_msg:
            raise SystemExit(f"第 {i} 轮模型未返回有效回答，请检查 LLM 配置。")

        _print_turn_header(i, user_msg, assistant_msg)
        _append_history(history, user_msg, assistant_msg)
        qa_pairs.append((user_msg, assistant_msg))

        turn_id = f"demo-{uuid.uuid4().hex[:12]}"
        ok = await record_episode(
            user_id=user_id,
            course_id=args.course_id,
            session_id=session_id,
            turn_id=turn_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            mode="chat",
            tools_used=tuple(model_out.get("tools_used") or ()),
        )
        turn_records.append({"round": i, "turn_id": turn_id, "recorded": ok, "model": model_out})
        print(f"\n>>> record_episode: {'成功' if ok else '跳过/失败'} (turn_id={turn_id})")

    print("\n>>> 开始 consolidate（按 session 批量升格到 mem0）...")
    async with AsyncSessionLocal() as db:
        result = await consolidate(db, user_id, args.course_id)
    print(f">>> consolidate 结果: {result}")

    after = await _fetch_all_memories(user_id, args.course_id)
    _print_block("【consolidate 后】mem0 全量记忆", after)

    search_ctx = ""
    if args.search_query:
        from core.memory.mem0_client import build_memory_context

        search_ctx = await build_memory_context(
            user_id, args.search_query, course_id=args.course_id, top_k=5
        )
        _print_block(f"【检索注入】query={args.search_query!r}", search_ctx or "(空)")

    return {
        "mode": "consolidate",
        "user_id": user_id,
        "course_id": args.course_id,
        "session_id": session_id,
        "qa_pairs": [{"user": u, "assistant": a} for u, a in qa_pairs],
        "turn_records": turn_records,
        "consolidate_result": result,
        "before_count": len(before),
        "after_count": len(after),
        "final_memories": after,
        "search_context": search_ctx,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from core.db.database import close_db, init_db

    await init_db()
    try:
        user_id = await _resolve_user_id(user_id=args.user_id, username=args.username)

        if args.cleanup_before:
            await _delete_all_memories(user_id)

        if args.mode == "consolidate":
            summary = await _run_consolidate(args, user_id)
        else:
            summary = await _run_direct(args, user_id)

        if args.cleanup_after:
            await _delete_all_memories(user_id)
            summary["cleaned_up_after"] = True

        return summary
    finally:
        await close_db()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="学生连续提问 + 模型真实回答 + 展示 mem0 存储")
    p.add_argument("--user-id", help="mem0 user_id（默认 demo-mem0-xxx）")
    p.add_argument("--username", help="使用数据库已有用户")
    p.add_argument("--course-id", default="demo-circuit-lab", help="课程 ID")
    p.add_argument(
        "--mode",
        choices=("direct", "consolidate"),
        default="direct",
        help="direct=每轮 mem0.add；consolidate=record_episode→consolidate",
    )
    p.add_argument("--turns", type=int, default=len(STUDENT_QUESTIONS), help="提问轮数")
    p.add_argument("--session-id", help="固定 session_id")
    p.add_argument(
        "--tools",
        default="rag",
        help="启用的工具，逗号分隔，如 rag,web_search；空串=不启用工具",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="等价于 --tools ''（不调 RAG，只测 mem0 + 纯模型回答）",
    )
    p.add_argument("--rag-mode", default="auto", help="RAG 模式：auto/naive/mix/llamaindex_pg 等")
    p.add_argument(
        "--search-query",
        default="并联电阻怎么算？我之前的错误有哪些？",
        help="结束后演示 build_memory_context 检索",
    )
    p.add_argument("--cleanup-before", action="store_true", help="运行前清空该 user 的 mem0")
    p.add_argument("--cleanup-after", action="store_true", help="运行后清空")
    p.add_argument("--out", type=argparse.FileType("w", encoding="utf-8"), help="写入 JSON 报告")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.no_rag:
        args.tools = ""
    args.enabled_tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    if args.turns < 1 or args.turns > len(STUDENT_QUESTIONS):
        sys.exit(f"错误: --turns 须在 1..{len(STUDENT_QUESTIONS)} 之间")

    logger.info(
        "模式=%s course=%s turns=%d tools=%s",
        args.mode, args.course_id, args.turns, args.enabled_tools or "(none)",
    )
    summary = asyncio.run(_run(args))

    print("\n" + "=" * 72)
    print("【总结】mem0 存的是什么？")
    print("=" * 72)
    print(
        f"  用户: {summary['user_id']}\n"
        f"  课程: {summary['course_id']}\n"
        f"  模式: {summary['mode']}\n"
        f"  记忆条数: {summary['before_count']} → {summary['after_count']}\n"
    )
    print("  mem0 每条记忆是 LLM 从「学生问 + 模型答」里抽取的短事实：")
    for item in summary.get("final_memories") or []:
        print(f"    - {item.get('memory')}")

    if args.out:
        json.dump(summary, args.out, ensure_ascii=False, indent=2, default=_json_default)
        args.out.close()
        logger.info("报告已写入 %s", args.out.name)


if __name__ == "__main__":
    main()
