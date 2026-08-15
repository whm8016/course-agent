"""仿真一轮（或多轮）问答，触发 L3 记忆沉淀并打印落库快照。

走与线上一致的主路径：record_episode → consolidate（mem0 + extract + 双图谱 + mastery）。

Usage（在 backend/ 目录，已配置 .env 与 PostgreSQL）:
    python -m scripts.simulate_memory_round --username YOUR_USER --course-id mycourse
    python -m scripts.simulate_memory_round --user-id abc123 --course-id mycourse --rounds 2
    python -m scripts.simulate_memory_round --username YOUR_USER --course-id mycourse --mock-extract
    python -m scripts.simulate_memory_round --username YOUR_USER --course-id mycourse --mock-extract --skip-mem0

说明:
  - 默认调真实 LLM 做 extract_knowledge；需 LLM__API_KEY。
  - --mock-extract: 不调 LLM，注入固定电路实验抽取 JSON（只看 graph/mastery 落库时用）。
  - --skip-mem0: mem0.add 改为 no-op 成功（mem0 未配时 graph/mastery 仍可跑）。
  - consolidate 在 mem0 失败时会跳过 graph/mastery，故本地 mem0 不可用时请加 --skip-mem0。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# 电路实验场景：含薄弱点 + 错误模式 + prerequisite 边（mock 模式用）
MOCK_EXTRACT_ROUND1: dict[str, list] = {
    "knowledge_points": [
        {
            "label": "电阻并联",
            "entity_id": "kp_parallel_resistor",
            "mastery_delta": -0.15,
            "risk_delta": 0.25,
            "notes": "把并联等效电阻算成相加",
            "relations": [{"target": "欧姆定律", "type": "prerequisite"}],
        },
        {
            "label": "戴维南等效",
            "entity_id": "kp_thevenin",
            "mastery_delta": -0.1,
            "risk_delta": 0.2,
            "notes": "开路电压与等效电阻混淆",
            "relations": [{"target": "叠加定理", "type": "prerequisite"}],
        },
    ],
    "error_patterns": [
        {
            "label": "并联电阻按串联相加",
            "severity": 0.75,
            "repeated": False,
            "notes": "1/R_eq = 1/R1 + 1/R2 误写为 R1+R2",
            "related_knowledge_points": ["电阻并联"],
            "correction_suggestions": ["先画等效电路，再列并联公式"],
        }
    ],
}

MOCK_EXTRACT_ROUND2: dict[str, list] = {
    "knowledge_points": [
        {
            "label": "电阻并联",
            "entity_id": "kp_parallel_resistor",
            "mastery_delta": -0.05,
            "risk_delta": 0.15,
            "notes": "第二次仍算错并联",
        },
    ],
    "error_patterns": [
        {
            "label": "并联电阻按串联相加",
            "severity": 0.8,
            "repeated": True,
            "notes": "复发：实验五前再次混淆",
            "related_knowledge_points": ["电阻并联"],
            "correction_suggestions": ["对比串联与并联公式"],
        }
    ],
}

DEFAULT_USER_MSG = (
    "老师，我在做实验三测并联电阻，我把两个 100Ω 并联算成了 200Ω，"
    "是不是应该相加？还有戴维南等效里开路电压和等效电阻我老是搞混。"
)

DEFAULT_ASSISTANT_MSG = (
    "并联电阻不能相加。两个相同阻值 R 并联，等效电阻是 R/2，即 100Ω 并联 100Ω 为 50Ω。"
    "公式是 1/R_eq = 1/R1 + 1/R2。戴维南等效分两步：先求开路电压 Uoc（端口开路），"
    "再求等效电阻 Req（独立源置零后从端口看入）。建议先掌握叠加定理再练戴维南。"
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"不可 JSON 序列化: {type(obj)!r}")


async def _resolve_user_id(db, *, user_id: str | None, username: str | None) -> str:
    from sqlalchemy import select

    from core.db.database import User

    if user_id:
        row = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
        if not row:
            raise SystemExit(f"用户不存在: user_id={user_id!r}")
        logger.info("使用用户 id=%s username=%s", row.id, row.username)
        return row.id
    if username:
        row = (await db.execute(select(User).where(User.username == username))).scalars().first()
        if not row:
            raise SystemExit(f"用户不存在: username={username!r}")
        logger.info("使用用户 id=%s username=%s", row.id, row.username)
        return row.id
    raise SystemExit("请指定 --user-id 或 --username")


async def _fetch_snapshot(db, user_id: str, course_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from core.db.database import KnowledgeMastery, MemoryEpisode, User
    from core.memory.graph_memory import load_graphs
    from core.memory.mastery import get_mastery_context

    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    episodes = (
        (
            await db.execute(
                select(MemoryEpisode)
                .where(
                    MemoryEpisode.user_id == user_id,
                    MemoryEpisode.course_id == course_id,
                )
                .order_by(MemoryEpisode.created_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    mastery_rows = (
        (
            await db.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == user_id,
                    KnowledgeMastery.course_id == course_id,
                )
            )
        )
        .scalars()
        .all()
    )
    kg, eg = await load_graphs(db, user_id)
    mastery_ctx = await get_mastery_context(db, user_id, course_id)

    return {
        "user": {
            "id": user.id if user else user_id,
            "username": user.username if user else None,
            "graph_version": user.graph_version if user else None,
        },
        "memory_episodes": [
            {
                "id": ep.id,
                "session_id": ep.session_id,
                "turn_id": ep.turn_id,
                "status": ep.status,
                "importance": ep.importance,
                "user_msg_preview": (ep.user_msg or "")[:120],
                "assistant_msg_preview": (ep.assistant_msg or "")[:120],
                "created_at": ep.created_at,
                "consolidated_at": ep.consolidated_at,
            }
            for ep in episodes
        ],
        "knowledge_mastery": [
            {
                "kp_id": r.kp_id,
                "label": r.label,
                "mastery": r.mastery,
                "risk": r.risk,
                "observation_count": r.observation_count,
                "last_observed_at": r.last_observed_at,
                "evidence_episode_ids": r.evidence_episode_ids,
            }
            for r in mastery_rows
        ],
        "knowledge_graph": kg,
        "error_graph": eg,
        "mastery_context_inject": mastery_ctx,
        "counts": {
            "episodes": len(episodes),
            "mastery_rows": len(mastery_rows),
            "kg_nodes": len(kg.get("nodes") or []),
            "kg_edges": len(kg.get("edges") or []),
            "eg_nodes": len(eg.get("nodes") or []),
            "eg_edges": len(eg.get("edges") or []),
        },
    }


def _print_snapshot(title: str, snap: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(json.dumps(snap, ensure_ascii=False, indent=2, default=_json_default))


@asynccontextmanager
async def _consolidate_patches(*, mock_extract: bool, skip_mem0: bool, extract_sequence: list[dict]):
    """mock_extract / skip_mem0 时 patch consolidate 依赖。"""
    patches: list[Any] = []
    if skip_mem0:
        patches.append(
            patch(
                "core.memory.mem0_client.get_memory",
                return_value=type("M", (), {"add": AsyncMock(return_value={"results": []})})(),
            )
        )
    if mock_extract:
        side_effects = list(extract_sequence) if extract_sequence else [MOCK_EXTRACT_ROUND1]

        async def _fake_extract(course_id: str, user_message: str, assistant_answer: str):
            idx = min(getattr(_fake_extract, "_i", 0), len(side_effects) - 1)
            _fake_extract._i = idx + 1
            payload = side_effects[idx]
            logger.info(
                "[mock-extract] course=%s round=%d kp=%d err=%d",
                course_id,
                idx + 1,
                len(payload.get("knowledge_points") or []),
                len(payload.get("error_patterns") or []),
            )
            return payload

        _fake_extract._i = 0
        patches.append(patch("core.memory.graph_memory.extract_knowledge", side_effect=_fake_extract))

    if not patches:
        yield
        return

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from core.memory.consolidation import consolidate
    from core.memory.episodic import record_episode

    await init_db()
    session_id = args.session_id or f"sim-{uuid.uuid4().hex[:10]}"
    extract_seq = [MOCK_EXTRACT_ROUND1, MOCK_EXTRACT_ROUND2] if args.rounds >= 2 else [MOCK_EXTRACT_ROUND1]

    try:
        async with AsyncSessionLocal() as db:
            user_id = await _resolve_user_id(db, user_id=args.user_id, username=args.username)
            before = await _fetch_snapshot(db, user_id, args.course_id)
            _print_snapshot(f"【沉淀前】user={user_id} course={args.course_id}", before)

        turn_results: list[dict[str, Any]] = []
        for i in range(args.rounds):
            turn_id = f"sim-{uuid.uuid4().hex[:12]}"
            user_msg = args.user_msg if i == 0 else (
                "我又把并联电阻算成相加了，实验五前还是搞不清楚。"
            )
            assistant_msg = args.assistant_msg if i == 0 else (
                "这是第二次出现同一类错误。请记住并联用倒数相加，串联才是直接相加。"
            )

            ok = await record_episode(
                user_id=user_id,
                course_id=args.course_id,
                session_id=session_id,
                turn_id=turn_id,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                mode="chat",
                tools_used=("rag",),
            )
            importance = None
            if ok:
                from core.memory.episodic import estimate_importance

                importance = estimate_importance(
                    user_message=user_msg,
                    agent_output=assistant_msg,
                    mode="chat",
                    tools_used=("rag",),
                )
            logger.info(
                "record_episode round=%d ok=%s session=%s turn=%s importance=%s",
                i + 1,
                ok,
                session_id,
                turn_id,
                importance,
            )
            turn_results.append(
                {"round": i + 1, "turn_id": turn_id, "recorded": ok, "importance": importance}
            )

        async with _consolidate_patches(
            mock_extract=args.mock_extract,
            skip_mem0=args.skip_mem0,
            extract_sequence=extract_seq[: args.rounds],
        ):
            async with AsyncSessionLocal() as db:
                logger.info("开始 consolidate user=%s course=%s ...", user_id, args.course_id)
                result = await consolidate(db, user_id, args.course_id)
                logger.info("consolidate 结果: %s", result)

        async with AsyncSessionLocal() as db:
            after = await _fetch_snapshot(db, user_id, args.course_id)
            _print_snapshot(f"【沉淀后】user={user_id} course={args.course_id}", after)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "course_id": args.course_id,
            "turns": turn_results,
            "consolidate_hint": (
                "mock-extract" if args.mock_extract else "live-llm-extract",
                "skip-mem0" if args.skip_mem0 else "mem0-live",
            ),
            "before_counts": before.get("counts"),
            "after_counts": after.get("counts"),
            "after": after,
        }
    finally:
        await close_db()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="仿真问答并观测 L3 记忆落库")
    p.add_argument("--user-id", help="users.id（与 --username 二选一）")
    p.add_argument("--username", help="users.username（与 --user-id 二选一）")
    p.add_argument("--course-id", required=True, help="课程 ID，如 mycourse")
    p.add_argument("--session-id", help="固定 session_id（默认每次随机 sim-xxx）")
    p.add_argument("--rounds", type=int, default=1, help="仿真轮数（2 轮可观察错误复发 repeated）")
    p.add_argument("--user-msg", default=DEFAULT_USER_MSG, help="学生提问文本")
    p.add_argument("--assistant-msg", default=DEFAULT_ASSISTANT_MSG, help="助手回答文本")
    p.add_argument(
        "--mock-extract",
        action="store_true",
        help="不调 LLM，注入固定抽取 JSON（推荐本地先看 graph/mastery）",
    )
    p.add_argument(
        "--skip-mem0",
        action="store_true",
        dest="skip_mem0",
        help="mem0.add 改为 no-op（mem0 未配时必须加，否则 graph/mastery 不会跑）",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="将沉淀后快照写入 JSON 文件",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.user_id and not args.username:
        sys.exit("错误: 请指定 --user-id 或 --username")
    if args.rounds < 1:
        sys.exit("错误: --rounds 至少为 1")
    if args.mock_extract:
        logger.info("模式: mock-extract（固定电路实验 JSON）")
    else:
        logger.info("模式: live extract_knowledge（需 LLM__API_KEY）")
    if args.skip_mem0:
        logger.info("mem0: skip（no-op）")
    else:
        logger.info("mem0: live（需 DB + mem0 可用；失败则 graph/mastery 被跳过）")

    summary = asyncio.run(_run(args))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        logger.info("快照已写入 %s", args.out.resolve())

    bc, ac = summary["before_counts"], summary["after_counts"]
    print("\n--- 变化摘要 ---")
    print(
        f"  episodes: {bc['episodes']} -> {ac['episodes']} | "
        f"mastery: {bc['mastery_rows']} -> {ac['mastery_rows']} | "
        f"kg_nodes: {bc['kg_nodes']} -> {ac['kg_nodes']} | "
        f"eg_nodes: {bc['eg_nodes']} -> {ac['eg_nodes']}"
    )
    if ac["mastery_rows"] == bc["mastery_rows"] and ac["kg_nodes"] == bc["kg_nodes"]:
        print(
            "  注意: 图谱/掌握度无变化——检查 consolidate 日志；mem0 失败时需加 --skip-mem0；"
            "或加 --mock-extract 排除 LLM 问题。"
        )


if __name__ == "__main__":
    main()
