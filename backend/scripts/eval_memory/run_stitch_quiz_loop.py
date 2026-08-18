r"""门控 ↔ 出题闭环对账（live）：门控说该补哪个缺口，出的题有没有打中它。

对 ``STITCH_CASES_LIVE`` 的 ``gold.stitch=True`` 正例：暴露轮（真 ChatPipeline →
真巩固，与 run_stitch_e2e_live 同一条写路径）之后，用同一学生档案调生产
``QuizPipeline.run("针对该生薄弱点出 1 道练习")``，程序判定生成题目的定向主题
与该 case 的 ``gold.evidence_topic`` 是否一致。

程序判定（零 LLM 评委，保持「评测方案_无学生离线」的零标注边界）：
  weak_hit    : gold.evidence_topic ∈ get_weak_topics 清单（出题定向的前提——
                缺口没进 top5 清单时 plan 无从定向，是前提失败不是生成失败）
  targeted_hit: 题目 target_topic_id == gold.evidence_topic（plan 定向直中）
  resolved_hit: target_topic_id 为空时，topic 文本经生产 match_topics 解析出的
                topic_id == gold.evidence_topic（兜底口径，评「打中了」不评「指名了」）
  aligned     : targeted_hit or resolved_hit —— 汇总进 quiz_gap_alignment_rate

明确不做：不模拟学生答题、不上 LLM 教学质量评委——只多问一句「出的题有没有
打中门控说的那个缺口」。

Usage（docker compose exec backend，现网栈+真实凭据）:
    python -m scripts.eval_memory.run_stitch_quiz_loop                 # 全部正例（19 例）
    python -m scripts.eval_memory.run_stitch_quiz_loop --case-id pos_recurrence_thevenin
    python -m scripts.eval_memory.run_stitch_quiz_loop --skip-expose   # 复用已种好的档案（调试定向逻辑）

⚠️ 同 run_stitch_e2e_live：全量跑之前先 ``docker compose stop worker``（拨钟 episode
会被 ARQ cron 当积压重判、双写观测污染档案），跑完 start。

费用（事先说清）：每例 ≈ 2-3 轮暴露（检索+mem0+聊天+判定 LLM）+ 1 次出题
（explore/plan/quiz 生成/critic 四轮 LLM）。19 正例 ≈ 48 暴露轮 + 19 次出题。
--skip-expose 跳过暴露只出题（档案复用上次跑出的，适合调 plan 定向问题）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# 现网栈凭据/DB 由容器 env 提供——live 层禁止 sqlite 内存库与 sk-test 假凭据
# （口径同 run_stitch_e2e_live，那套 setdefault 不搬）。
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from .run_stitch_e2e_live import (  # noqa: E402
    COURSE_ID,
    _CLOCK_DAYS_MAX,
    _cleanup_user,
    _expose_case,
    _gate_fields,
    _probe_retrieval,
)
from .stitch_cases_table_e_live import STITCH_CASES_LIVE, live_case_now  # noqa: E402

# 出题要求（固定口径，别按 case 改——评的是生产管线面对这句话的定向能力）
_QUIZ_REQUIREMENT = "针对该生薄弱点出 1 道练习"


def _judge_alignment(target: str, resolved_fallback: str, gold: str) -> dict:
    """纯函数：定向判定（targeted/resolved/aligned）。

    target 非空只走 targeted 口径（plan 指名了主题就以指名为准，不再拿 topic 文本
    兜底——两个口径混算会把「指名打偏但文本碰巧命中」记成对）。target 为空才用
    match_topics 解析 topic 文本的 resolved 口径。
    """
    targeted = bool(target) and target == gold
    resolved = (not target) and bool(resolved_fallback) and resolved_fallback == gold
    return {
        "targeted_hit": targeted,
        "resolved_hit": resolved,
        "aligned": targeted or resolved,
    }


async def _run_quiz(user_id: str, course_id: str) -> dict:
    """生产 QuizPipeline 出 1 道练习（explore/plan/quiz 全真，含 critic 校验轮）。"""
    from core.context import UnifiedContext
    from core.question.pipeline import QuizPipeline
    from core.stream_bus import StreamBus

    ctx = UnifiedContext(
        user_message=_QUIZ_REQUIREMENT,
        user_id=user_id,
        course_id=course_id,
        enabled_tools=["rag"],
    )
    bus = StreamBus()
    try:
        return await QuizPipeline().run(_QUIZ_REQUIREMENT, ctx, bus, count=1)
    finally:
        await bus.close()


async def run(case_ids: list[str] | None = None, *, skip_expose: bool = False) -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from core.memory.mastery import get_weak_topics
    from core.memory.topic_match import match_topics

    await _probe_retrieval(COURSE_ID)
    logger.warning(
        "[quiz-loop] 运行前提：先 docker compose stop worker 再跑（同 run_stitch_e2e_live，"
        "拨钟 episode 会被 ARQ cron 重判双写观测）；跑完 docker compose start worker"
    )

    known = {c["id"] for c in STITCH_CASES_LIVE}
    if case_ids:
        missing = set(case_ids) - known
        if missing:
            raise SystemExit(f"未知 case_id: {sorted(missing)}")
    selected = set(case_ids) if case_ids else None

    wall0 = time.time()
    base = wall0 - _CLOCK_DAYS_MAX * 86400.0
    rows: list[dict] = []

    await init_db()
    try:
        for idx, c in enumerate(STITCH_CASES_LIVE):
            # 只对正例出题：负例门控本就不该提，出题对齐无从谈起（表E 已评门控侧）
            if not c["gold"]["stitch"]:
                continue
            if selected is not None and c["id"] not in selected:
                continue
            # 独立 user 前缀 lvq_：别复用 live 表E 的 lv_（那边的冷却戳/观测会互相污染）
            user_id = f"lvq_{idx:02d}"
            case_base = (
                wall0
                if ("cooldown_hours" in c and "clock_days" not in c)
                else base
            )
            logger.info("[quiz-loop] === %s user=%s ===", c["id"], user_id)

            from .seed_stitch_personas import _ensure_user

            async with AsyncSessionLocal() as db:
                await _ensure_user(db, user_id)
                await _cleanup_user(db, user_id, COURSE_ID)

            if skip_expose:
                logger.info("[quiz-loop] %s --skip-expose：复用库内档案", c["id"])
            else:
                await _expose_case(user_id, COURSE_ID, c, case_base)

            gold_topic = c["gold"].get("evidence_topic") or ""
            async with AsyncSessionLocal() as db:
                # 门控信息列（本脚本不重评门控，只记录「当时门控怎么说」）
                gate_brief = await _probe_gate_info(db, user_id, c, base)
                weak_topics = await get_weak_topics(db, user_id, COURSE_ID)
                weak_hit = any(w["topic_id"] == gold_topic for w in weak_topics)

            result = await _run_quiz(user_id, COURSE_ID)
            questions = result.get("questions") or []
            q = questions[0] if questions else {}
            target = str(q.get("target_topic_id") or "").strip()
            resolved_fallback = ""
            if not target and q.get("topic"):
                async with AsyncSessionLocal() as db:
                    hits = await match_topics(str(q["topic"]), COURSE_ID, db)
                resolved_fallback = hits[0][0] if hits else ""

            verdict = _judge_alignment(target, resolved_fallback, gold_topic)
            row = {
                "id": c["id"], "label": c["label"], "user_id": user_id,
                "gold_evidence": gold_topic,
                "gate": gate_brief,
                "weak_hit": weak_hit,
                "weak_topics": weak_topics,
                "n_questions": len(questions),
                "question_topic": q.get("topic", ""),
                "target_topic_id": target,
                "resolved_fallback": resolved_fallback,
                **verdict,
                "quiz_metadata": {
                    k: result.get("metadata", {}).get(k)
                    for k in (
                        "weak_topics_considered", "weak_topics_targeted",
                        "critic_passed", "critic_regenerated", "critic_failed",
                        "count_failed",
                    )
                },
            }
            rows.append(row)
            logger.info(
                "[quiz-loop] %s gold=%s weak_hit=%s target=%s aligned=%s",
                c["id"], gold_topic, weak_hit, target or resolved_fallback or "∅",
                verdict["aligned"],
            )
    finally:
        await close_db()

    n = len(rows)
    hit = sum(1 for r in rows if r["aligned"])
    return {
        "suite": "stitch_quiz_gap_alignment",
        "cases": n,
        "base": base,
        "base_utc": datetime.fromtimestamp(base, timezone.utc).isoformat(),
        "quiz_gap_alignment_rate": round(hit / n, 4) if n else None,
        # 分解指标：前提（缺口进清单）/ 直中 / 文本兜底——错在哪一层一眼看穿
        "weak_prereq_rate": round(sum(1 for r in rows if r["weak_hit"]) / n, 4) if n else None,
        "targeted_rate": round(sum(1 for r in rows if r["targeted_hit"]) / n, 4) if n else None,
        "resolved_rate": round(sum(1 for r in rows if r["resolved_hit"]) / n, 4) if n else None,
        "no_question_cases": sum(1 for r in rows if r["n_questions"] == 0),
        "details": rows,
    }


async def _probe_gate_info(db, user_id: str, c: dict, base: float) -> dict:
    """信息列：拨钟探测一次门控（不 commit 冷却——本脚本不评冷却，别烧冷却戳）。"""
    from core.memory.proactive import stitch_for_turn

    brief = await stitch_for_turn(c["query"], user_id, COURSE_ID, db, now=live_case_now(c, base))
    return _gate_fields(brief)


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser(
        description="门控↔出题闭环对账：暴露轮触发门控后出 1 题核对定向（需现网栈+真实凭据）")
    parser.add_argument("--case-id", action="append", help="只跑指定 case（可多次给）")
    parser.add_argument("--skip-expose", action="store_true",
                        help="跳过暴露轮复用库内档案（调试 plan 定向逻辑用）")
    parser.add_argument("--out", type=Path,
                        help="输出 JSON（默认 results/stitch_quiz_gap_alignment.json）")
    args = parser.parse_args()

    summary = asyncio.run(run(case_ids=args.case_id, skip_expose=args.skip_expose))
    out = args.out or config.RESULTS_DIR / "stitch_quiz_gap_alignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "[quiz-loop] %d 例 alignment=%.4f -> %s",
        summary["cases"], summary["quiz_gap_alignment_rate"] or 0.0, out,
    )


if __name__ == "__main__":
    main()
