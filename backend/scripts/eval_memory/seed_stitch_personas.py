r"""为 mycourse 门控评测「造学生」：真实多轮对话 → record_episode → consolidate 沉淀学情。

**不**直接写 ``knowledge_mastery`` / ``error_graph``——走与线上一致的：
  ChatPipeline（真实 LLM 回答）→ record_episode → consolidate（mem0 + extract + graph + mastery）

每个学生 persona 对应 ``stitch_cases_mycourse`` 一个 case 的学情目标；对话脚本只预设
**学生提问**（含真实课程词汇，便于 extract 对齐 course_topic label），助手回答由模型生成。

产物：``results/stitch_personas_<course_id>.json``（case_id → user_id + 沉淀快照 + 门控验真）

Usage（docker exec backend）:
    python -m scripts.eval_memory.seed_stitch_personas
    python -m scripts.eval_memory.seed_stitch_personas --case-id pos_prereq_equiv_r
    python -m scripts.eval_memory.seed_stitch_personas --cleanup-before

沉淀完成后跑三臂 e2e（用真实用户，不再 _seed_case_profile）:
    python -m scripts.eval_memory.run_stitch_e2e --suite mycourse --seeded-users results/stitch_personas_mycourse.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# user_id / session_id 须 ≤32 字符（memory_episodes FK）
CASE_USER_IDS: dict[str, str] = {
    "pos_prereq_equiv_r": "ss-01",
    "pos_prereq_thevenin": "ss-02",
    "pos_prereq_potentiometer": "ss-03",
    "pos_recurrence_thevenin": "ss-04",
    "neg_unknown": "ss-05",
    "neg_irrelevant": "ss-06",
    "neg_corrected": "ss-07",
    "neg_all_low": "ss-08",
    "neg_stage_mismatch": "ss-09",
}


def _user_id_for_case(case_id: str) -> str:
    uid = CASE_USER_IDS.get(case_id)
    if not uid:
        raise ValueError(f"无 CASE_USER_IDS 映射: {case_id}")
    return uid

# 每个 persona：多 session × 多轮学生问句（助手答由 ChatPipeline 实时生成）
# 问句刻意带上讲义知识点用语，帮助 extract_knowledge / mastery label 对齐 course_topic。
PERSONA_DIALOGUES: dict[str, list[list[str]]] = {
    "pos_prereq_equiv_r": [
        [
            "老师，实验二测线性电阻元件的伏安特性，我描的点不成直线，是不是表接错了？",
            "我还是搞不清线性电阻两端电压和电流的关系，算出来的电阻每次都不一样。",
            "线性电阻伏安特性应该是直线吧？我第三次复测还是弯的，肯定是我不会测。",
        ],
        [
            "做完实验二我又把线性电阻和欧姆定律搞混了，U=IR 我都会背但不会用。",
            "线性电阻元件的伏安特性我复习了两遍还是不懂斜率代表什么，实验报告写不出分析。",
        ],
    ],
    "pos_prereq_thevenin": [
        [
            "做实验五前，我对实际电压源的伏安特性还是糊的，外特性曲线看不懂。",
            "独立源置零是什么意思？求戴维南等效电阻时电压源怎么处理？",
            "实际电压源和理想电压源的区别我还是分不清，实验四的数据我也解释不了。",
        ],
        [
            "我又把实际电压源的内阻当成外接电阻了，伏安特性图还是画不对。",
        ],
    ],
    "pos_prereq_potentiometer": [
        [
            "实验六里三端变阻器我有三种接法，不知道哪种是分压器，滑动端接哪里？",
            "分压器调压特性曲线我画出来是弯的，是不是数据取错了？",
            "分压器原理我看书两遍还是不懂，滑动端接错是不是就测不到分压？",
        ],
    ],
    "pos_recurrence_thevenin": [
        [
            "戴维南定理我还是会把开路电压 Uoc 和等效电阻 Req 搞混，步骤能再说一遍吗？",
        ],
        [
            "又错了：求戴维南等效时我把短路电流当成开路电压了，这是不是老毛病？",
            "戴维南等效电路外特性验证报告里要画什么图？我上次也是 Uoc 和 Req 混着算。",
        ],
    ],
    "neg_unknown": [
        [
            "实验报告封面上的学号填错了可以重交吗？",
            "下周五实验课调课通知在哪里看？",
        ],
    ],
    "neg_irrelevant": [
        [
            "实验十一滤波器设计里串并联谐振参数我算不对，LC 取值怎么估？",
            "滤波器实验波形观测报告格式有模板吗？",
        ],
    ],
    "neg_corrected": [
        [
            "线性电阻元件伏安特性测量我一开始表接错了，后来改对了，数据是线性的了。",
            "按你上次说的方法重新测了线性电阻，斜率就是电阻值，这次应该对了。",
        ],
    ],
    "neg_all_low": [
        [
            "线性电阻伏安特性我这次测量很顺利，数据点基本在一条直线上。",
            "实际电压源和理想电压源的伏安特性区别我理解了，独立源置零也会了。",
            "戴维南定理求 Uoc 和 Req 的步骤我按流程做了一遍，应该没问题。",
        ],
    ],
    "neg_stage_mismatch": [
        [
            "分压变阻器阻值与容量选取我老是选太大，实验六报告被扣分了。",
            "分压器技术要求里额定电流怎么估算？",
        ],
    ],
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


async def _ensure_user(db, user_id: str, case_id: str) -> None:
    from core.db.database import User
    from sqlalchemy import select

    row = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if row is None:
        db.add(
            User(
                id=user_id,
                username=user_id,
                password_hash="seed",
                role="student",
            )
        )
        await db.commit()


async def _clear_user_course_data(db, user_id: str, course_id: str) -> None:
    from core.db.database import KnowledgeMastery, MemoryEpisode, User
    from sqlalchemy import delete, select

    await db.execute(
        delete(KnowledgeMastery).where(
            KnowledgeMastery.user_id == user_id, KnowledgeMastery.course_id == course_id
        )
    )
    await db.execute(
        delete(MemoryEpisode).where(
            MemoryEpisode.user_id == user_id, MemoryEpisode.course_id == course_id
        )
    )
    row = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if row:
        row.error_graph = {"nodes": [], "edges": []}
        row.knowledge_graph = {"nodes": [], "edges": []}
    await db.commit()


async def _fetch_mastery_snapshot(db, user_id: str, course_id: str, case_id: str) -> dict[str, Any]:
    from core.db.database import KnowledgeMastery
    from core.memory.course_topic_store import _norm_label, get_course_topic_map, get_prereq_predecessors
    from core.memory.graph_memory import load_graphs
    from core.memory.proactive import (
        _read_error_repeated, _read_mastery, decide_stitch, stitch_for_turn,
    )
    from sqlalchemy import select

    from .stitch_cases_mycourse import STITCH_CASES

    case = next(c for c in STITCH_CASES if c["id"] == case_id)

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
    topic_map = await get_course_topic_map(course_id, db)

    matched = case["matched_topic"]
    closure = await get_prereq_predecessors(matched, course_id, db) if matched else set()
    # 门控读侧走真实 _read_mastery（logit = theta - b_全班，与线上同口径）；
    # C2 复发走真实 _read_error_repeated（观测日志 poor 计数），不再读 error_graph
    mastery_map = await _read_mastery(user_id, course_id, db, topic_map)
    error_repeated = await _read_error_repeated(user_id, course_id, db, topic_map)

    brief = decide_stitch(matched, closure, mastery_map, error_repeated)
    gate_brief = await stitch_for_turn(case["query"], user_id, course_id, db)

    return {
        "case_id": case_id,
        "knowledge_mastery": [
            {
                "kp_id": r.kp_id,
                "label": r.label,
                "mastery": r.mastery,
                "risk": r.risk,
                "observation_count": r.observation_count,
                "theta": r.theta,
                "n_obs": r.n_obs,
                "topic_id_resolved": topic_map.get(_norm_label(r.label)),
            }
            for r in mastery_rows
        ],
        "error_graph_nodes": (eg or {}).get("nodes") or [],
        "gate_decision": {
            "query": case["query"],
            "gold_stitch": case["gold"]["stitch"],
            "decide_stitch": brief.should_stitch,
            "decide_reason": brief.reason,
            "stitch_for_turn": gate_brief.should_stitch,
            "stitch_reason": gate_brief.reason,
            "st_match_ok": brief.should_stitch == case["gold"]["stitch"],
        },
    }


async def _backfill_mastery_from_episodes(db, user_id: str, course_id: str) -> int:
    """掌握度补跑兜底：consolidate 判定失败（超预算/超时）的 episode 留 pending，
    这里对仍 pending 的段重跑一次 consolidate——走**与线上同一条** judge → 观测 →
    Elo 折叠路径。

    旧实现走 extract_knowledge + append_mastery（delta 混合 + label 哈希 kp_id），
    那条路径已被掌握度估计器重构否掉：种出的行 theta/n_obs 全 0、kp_id 非 topic_id，
    门控读侧（logit + topic_id 对齐）读不懂，正例全变 FN。已删除。
    """
    from core.memory.consolidation import consolidate
    from sqlalchemy import func, select

    from core.db.database import MemoryEpisode

    n_pending = (
        await db.execute(
            select(func.count()).select_from(MemoryEpisode).where(
                MemoryEpisode.user_id == user_id,
                MemoryEpisode.course_id == course_id,
                MemoryEpisode.status == "pending",
            )
        )
    ).scalar_one()
    if not n_pending:
        return 0

    result = await consolidate(db, user_id, course_id)
    return result.get("promoted", 0)


async def _run_persona(
    case_id: str,
    course_id: str,
    *,
    no_rag: bool,
    cleanup_before: bool,
) -> dict[str, Any]:
    from core.db.database import AsyncSessionLocal
    from core.memory.consolidation import consolidate
    from core.memory.episodic import record_episode
    from scripts.demo_mem0_student_qa import _append_history, _ask_model

    sessions = PERSONA_DIALOGUES.get(case_id)
    if not sessions:
        raise ValueError(f"无对话脚本: {case_id}")

    user_id = _user_id_for_case(case_id)
    enabled_tools: list[str] = [] if no_rag else []
    history: list[dict[str, Any]] = []
    session_logs: list[dict[str, Any]] = []

    async with AsyncSessionLocal() as db:
        await _ensure_user(db, user_id, case_id)
        if cleanup_before:
            await _clear_user_course_data(db, user_id, course_id)

    for si, questions in enumerate(sessions, start=1):
        session_id = f"{user_id}-s{si}-{uuid.uuid4().hex[:4]}"
        turns: list[dict[str, Any]] = []
        logger.info("[seed] case=%s session=%d questions=%d", case_id, si, len(questions))

        for qi, user_msg in enumerate(questions, start=1):
            model_out = await _ask_model(
                user_id=user_id,
                course_id=course_id,
                session_id=session_id,
                user_msg=user_msg,
                history=history,
                enabled_tools=enabled_tools,
                rag_mode="auto",
            )
            assistant_msg = model_out["assistant_msg"]
            if not assistant_msg or assistant_msg.startswith("[模型错误]"):
                raise RuntimeError(f"case={case_id} session={si} q={qi} 模型无有效回答: {model_out}")

            turn_id = f"t-{uuid.uuid4().hex[:10]}"
            ok = await record_episode(
                user_id=user_id,
                course_id=course_id,
                session_id=session_id,
                turn_id=turn_id,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                mode="chat",
                tools_used=tuple(model_out.get("tools_used") or ()),
            )
            _append_history(history, user_msg, assistant_msg)
            turns.append({"q": qi, "user": user_msg, "recorded": ok, "turn_id": turn_id})
            print(f"  [{case_id} s{si} q{qi}] recorded={ok} 问: {user_msg[:60]}...")

        async with AsyncSessionLocal() as db:
            result = await consolidate(db, user_id, course_id)
        logger.info("[seed] consolidate case=%s session=%d → %s", case_id, si, result)
        session_logs.append({"session_id": session_id, "turns": turns, "consolidate": result})

    async with AsyncSessionLocal() as db:
        backfilled = await _backfill_mastery_from_episodes(db, user_id, course_id)
        if backfilled:
            logger.info("[seed] retry-consolidate mastery case=%s rows=%d", case_id, backfilled)

    async with AsyncSessionLocal() as db:
        snap = await _fetch_mastery_snapshot(db, user_id, course_id, case_id)

    flag = "✅" if snap["gate_decision"]["st_match_ok"] else "⚠️"
    print(
        f"{flag} {case_id} user={user_id} decide_stitch={snap['gate_decision']['decide_stitch']} "
        f"gold={snap['gate_decision']['gold_stitch']} mastery_rows={len(snap['knowledge_mastery'])}"
    )
    return {"case_id": case_id, "user_id": user_id, "sessions": session_logs, "snapshot": snap}


async def _main(args: argparse.Namespace) -> dict:
    from core.db.database import close_db, init_db
    from sqlalchemy import func, select

    from core.db.database import AsyncSessionLocal, CourseTopic
    from .stitch_cases_mycourse import COURSE_ID, STITCH_CASES

    course_id = args.course_id or COURSE_ID
    case_ids = [args.case_id] if args.case_id else [c["id"] for c in STITCH_CASES]

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            n = (
                await db.execute(
                    select(func.count()).select_from(CourseTopic).where(CourseTopic.course_id == course_id)
                )
            ).scalar_one()
        if not n:
            raise SystemExit(f"课程 {course_id} 无 course_topic，请先建图")

        personas: dict[str, Any] = {}
        for cid in case_ids:
            personas[cid] = await _run_persona(
                cid, course_id, no_rag=args.no_rag, cleanup_before=args.cleanup_before
            )

        out = {
            "course_id": course_id,
            "seeded_at": time.time(),
            "personas": personas,
            "user_map": {cid: p["user_id"] for cid, p in personas.items()},
        }
        return out
    finally:
        await close_db()


def main() -> None:
    p = argparse.ArgumentParser(description="真实对话沉淀门控评测用学生 persona")
    p.add_argument("--course-id", default="mycourse")
    p.add_argument("--case-id", help="只造指定 case 的 persona")
    p.add_argument("--no-rag", action="store_true", help="关闭 RAG，减少噪声")
    p.add_argument("--cleanup-before", action="store_true", help="重跑前清该 user 的本课 episode/mastery")
    p.add_argument("--out", type=Path, help="输出 JSON（默认 results/stitch_personas_<course>.json）")
    args = p.parse_args()

    summary = asyncio.run(_main(args))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or RESULTS_DIR / f"stitch_personas_{summary['course_id']}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    logger.info("[seed] 已写入 %s", out_path)
    print(f"\n下一步三臂 e2e:\n  python -m scripts.eval_memory.run_stitch_e2e --suite mycourse --seeded-users {out_path}")


if __name__ == "__main__":
    main()
