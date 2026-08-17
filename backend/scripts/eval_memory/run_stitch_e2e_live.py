r"""表E live 全栈评测：30 例走现网课全栈（真 ChatPipeline[RAG+mem0] → 巩固 → 拨钟门控）。

与离线 v2 层（run_stitch_table_e）/ 种档案 e2e 层（run_stitch_e2e）的区别：那两层
把学情档案种进库再读；本层**写路径也走生产**——学生问句由真实 ChatPipeline 回答
（enabled_tools=["rag"] + rag_mode="auto"，讲义预检索与 RAG 工具都在；mem0 读入
上下文）、record_episode（created_at=剧本钟）→ consolidate（真判定器 + mem0 升格），
探测轮只打生产读侧 stitch_for_turn（评 When 门）。**禁止 INSERT 观测装学生**——
评测必须打在写入口上（CQS：对话账本是发件箱，巩固是唯一写掌握度的入口）。

Usage（docker compose exec backend，现网栈+真实凭据）:
    python -m scripts.eval_memory.run_stitch_e2e_live --check-match   # 预检：30 探测问句 NN 匹配（零 LLM）
    python -m scripts.eval_memory.run_stitch_e2e_live                 # 全量 30 例（≈77 暴露轮，费用见下）
    python -m scripts.eval_memory.run_stitch_e2e_live --case-id cooldown_1h

时钟口径：
- 观测钟锚 base = 开跑墙钟 wall0 − 27 天（时钟三态 day28 的探测钟 = base+27d ≈
  wall0，见 _CLOCK_DAYS_MAX 注释）；全部遗忘/复发窗按注入钟实算，与墙钟无关。
- 冷却例（cooldown_hours 键）探测走**墙钟**：probe1 = time.time() → 若要拼则
  commit_stitch_cooldown（与 turn_runtime 同一函数，落墙钟）→ probe2 = probe1 +
  N 小时。不手写 last_stitched_at。

费用（事先说清）：每条暴露轮 ≈ 检索 + mem0 读 + 聊天 LLM（可能再调 rag 工具）+
问句向量 + 判定 LLM + 巩固里 mem0 写。30 例约 77 轮暴露 + 71 段判定，比关检索版贵。

**明确仍假的部分**（写在这里，避免下次再吵）：
- 学生问句是剧本，不是真人；
- 金标是重审后的教师规则句子（stitch_cases_table_e_live，逐句经用户点头），
  不是课堂 extra-hint 标签；
- 巩固是同步 await consolidate()，不是 ARQ 延迟任务；
- 探测轮只打门控（When），不打「简报有没有改变模型讲稿」（What/How 不评）；
- RAG / mem0 **开着**——它们是产品的一部分，不再列进「假的」；
- 冷却落盘的时间戳是墙钟（与线上一致），探测钟按 case 拨（见上）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 现网栈凭据/DB 由容器 env 提供——这里**故意不设** DB__URL / LLM__API_KEY 默认值：
# live 层禁止 sqlite 内存库与 sk-test 假凭据（run_stitch_e2e 的 setdefault 那套不搬）。
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from .stitch_cases_table_e_live import STITCH_CASES_LIVE, live_case_now  # noqa: E402
from .stitch_scripts_table_e import expose_sessions  # noqa: E402

COURSE_ID = "mycourse"

# 时钟三态最大偏移（27 天）。base = wall0 − 27d 使 clock_day28 两例的探测钟
# （base + 27d）≈ 开跑墙钟——clock_day28_cooldown 的冷却对账（commit 落墙钟）才对得上。
# 例外：冷却三例（cooldown_hours 且无 clock_days）的观测锚 **wall0 本身**——它们
# dsl=3+1 天间距，若也挪 27 天，墙钟探测时复发窗（30 天）恰好一条 poor 都装不下，
# C2 消失、probe1 不拼、冷却永远测不到（写 case 时实测踩过）。
_CLOCK_DAYS_MAX = 27

# 开跑前的现网课内探针问句（检索命中闸）：必须打到讲义块才证明索引挂上了
_PROBE_QUERY = "线性电阻元件的伏安特性怎么测量？"


def _outcome_level(outcome: float) -> str:
    """观测 outcome → 等级（LEVEL_OUTCOME 逆映射，poor<=0.25 / good>=0.75 / 其间 fair）。"""
    if outcome <= 0.25:
        return "poor"
    return "good" if outcome >= 0.75 else "fair"


async def _probe_retrieval(course_id: str) -> None:
    """凭据闸（fail-fast，表A 口径）：embedding 探针 + 一句课内问句真检索。

    检索零命中直接停——索引没挂上时暴露轮会静默退化成裸聊，30 例白烧。
    LLM 侧不单独探：第一轮 _ask_model 失败即 raise（fail-fast 同 seed 脚本）。"""
    from core.rag.llamaindex.pg_store import get_embed_model

    if (os.environ.get("LLM__API_KEY") or "") in ("", "sk-test"):
        raise SystemExit("LLM__API_KEY 缺失或为 sk-test——live 层要真实凭据，别拿假 key 评")
    try:
        model = get_embed_model()
        await model._aget_query_embedding("测试")
    except Exception as exc:
        raise SystemExit(f"embedding 探针失败（缺 EMBEDDING__API_KEY？）：{exc}") from exc

    from core.agent.tool_registry import execute_tool

    result = None
    try:
        # 与 chat_pipeline.retrieve_kb_seed 同一条检索路（fact 策略），探真不探假
        result = await execute_tool(
            "rag", course_id=course_id, user_id="eval-probe",
            query=_PROBE_QUERY, mode="auto", strategy="fact",
        )
    except Exception as exc:
        raise SystemExit(f"检索探针异常（索引/后端没挂上？）：{exc}") from exc
    text = str(result.content).strip() if (result and result.success) else ""
    if not text:
        raise SystemExit(
            f"检索探针零命中（query={_PROBE_QUERY!r}）——现网索引未就绪，"
            "先排查 KB/索引再跑，不许默默退化成裸聊"
        )
    logger.info("[live] 检索探针命中 %d 字——索引就绪", len(text))


async def _ensure_user(db, user_id: str) -> None:
    from core.db.database import User
    from sqlalchemy import select

    row = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if row is None:
        db.add(User(id=user_id, username=user_id, password_hash="seed", role="student"))
        await db.commit()


async def _cleanup_user(db, user_id: str, course_id: str) -> None:
    """重跑幂等：只清这个评测 user 的 episode / mastery / 观测（mem0 best-effort）。

    不碰 course_topic / 其他用户——隔离只靠 user_id（同生产读域），绝不改生产图。"""
    from core.db.database import KnowledgeMastery, MasteryObservation, MemoryEpisode
    from sqlalchemy import delete

    await db.execute(delete(MasteryObservation).where(
        MasteryObservation.user_id == user_id,
        MasteryObservation.course_id == course_id,
    ))
    await db.execute(delete(KnowledgeMastery).where(
        KnowledgeMastery.user_id == user_id,
        KnowledgeMastery.course_id == course_id,
    ))
    await db.execute(delete(MemoryEpisode).where(
        MemoryEpisode.user_id == user_id,
        MemoryEpisode.course_id == course_id,
    ))
    await db.commit()
    try:
        from core.memory.mem0_client import get_memory

        await get_memory().delete_all(user_id=user_id)
    except Exception as exc:
        logger.warning("[live] mem0 清理失败（非致命，继续）：%s", exc)


async def _expose_case(
    user_id: str, course_id: str, case: dict, base: float
) -> tuple[list[dict], int]:
    """跑暴露轮剧本：session 升序，每轮真 ChatPipeline → episode(拨钟) → 每 session 巩固。

    返回 (逐轮日志, 巩固重试次数)。判定失败留 pending 的段，末尾再 consolidate
    一次兜底（与 seed_stitch_personas._backfill_mastery_from_episodes 同思路）。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.consolidation import consolidate
    from core.memory.episodic import record_episode
    from scripts.demo_mem0_student_qa import _append_history, _ask_model

    history: list[dict[str, Any]] = []
    turn_logs: list[dict] = []
    consolidates = 0
    for si, sess in enumerate(expose_sessions(case, base), start=1):
        session_id = f"{user_id}-x{si}-{uuid.uuid4().hex[:4]}"[:32]
        at = sess["at"]
        logger.info(
            "[live] %s session=%d at=%s turns=%d",
            case["id"], si, datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%d"),
            len(sess["turns"]),
        )
        for ti, turn in enumerate(sess["turns"], start=1):
            model_out = await _ask_model(
                user_id=user_id,
                course_id=course_id,
                session_id=session_id,
                user_msg=turn["q"],
                history=history,
                enabled_tools=["rag"],
                rag_mode="auto",
                extra_metadata={"_eval_now": at},
            )
            assistant_msg = model_out["assistant_msg"]
            if not assistant_msg or assistant_msg.startswith("[模型错误]"):
                raise RuntimeError(
                    f"case={case['id']} session={si} turn={ti} 模型无有效回答: {model_out}"
                )
            ok = await record_episode(
                user_id=user_id,
                course_id=course_id,
                session_id=session_id,
                turn_id=f"t-{uuid.uuid4().hex[:10]}",
                user_msg=turn["q"],
                assistant_msg=assistant_msg,
                mode="chat",
                tools_used=tuple(model_out.get("tools_used") or ()),
                created_at=at,
            )
            _append_history(history, turn["q"], assistant_msg)
            turn_logs.append({
                "session": si, "topic_id": turn["topic_id"], "q": turn["q"],
                "recorded": ok, "tools_used": model_out.get("tools_used") or [],
                "assistant_chars": len(assistant_msg),
            })

        async with AsyncSessionLocal() as db:
            result = await consolidate(db, user_id, course_id)
        consolidates += 1
        logger.info("[live] %s session=%d consolidate → %s", case["id"], si, result)

    # 兜底重试：判定超时/超预算的段留 pending，重跑一次（走同一条 judge→观测→折叠路）
    async with AsyncSessionLocal() as db:
        retry = await consolidate(db, user_id, course_id)
    if retry.get("claimed"):
        logger.info("[live] %s 巩固兜底补跑 → %s", case["id"], retry)
    return turn_logs, (1 if retry.get("claimed") else 0)


async def _read_extract(db, user_id: str, course_id: str, case: dict) -> dict:
    """读真观测对账期望档案：extract_ok = 主题出现且末档=期望末档（评测信息量，不炸）。

    judge_levels 是「判定器长出来的世界」；打偏（多判/漏判/错档）全部记列，
    不从混淆矩阵里剔。"""
    from core.db.database import MasteryObservation
    from sqlalchemy import select

    rows = (
        await db.execute(
            select(MasteryObservation.topic_id, MasteryObservation.outcome,
                   MasteryObservation.created_at)
            .where(
                MasteryObservation.user_id == user_id,
                MasteryObservation.course_id == course_id,
            )
            .order_by(MasteryObservation.created_at)
        )
    ).all()
    by_topic: dict[str, list[dict]] = {}
    for tid, outcome, created_at in rows:
        by_topic.setdefault(tid, []).append(
            {"level": _outcome_level(float(outcome)), "at": created_at}
        )
    extract_topics = sorted(by_topic)
    detail: dict[str, dict] = {}
    n_ok = 0
    for m in case["mastery"]:
        tid = m["topic_id"]
        got = by_topic.get(tid, [])
        got_levels = [g["level"] for g in got]
        want = m["levels"]
        ok = bool(got_levels) and got_levels[-1] == want[-1]
        n_ok += int(ok)
        detail[tid] = {
            "expected": want, "actual": got_levels, "extract_ok": ok,
        }
    return {
        "extract_topics": extract_topics,
        "extract_detail": detail,
        "extract_ok": n_ok == len(case["mastery"]),
        "extract_ok_count": n_ok,
        "n_observations": len(rows),
    }


async def _probe_gate(db, user_id: str, course_id: str, case: dict, base: float) -> dict:
    """探测轮（只打门控）。冷却例走墙钟三步：probe1 → commit → probe2 记分。"""
    from core.memory.proactive import commit_stitch_cooldown, stitch_for_turn

    query = case["query"]
    cooldown_hours = case.get("cooldown_hours")
    if cooldown_hours is None:
        now = live_case_now(case, base)
        brief = await stitch_for_turn(query, user_id, course_id, db, now=now)
        return {
            "now": now, "gate": _gate_fields(brief),
            "cooldown_committed": False, "pre_gate": None,
        }

    t0 = time.time()  # 墙钟——commit_stitch_cooldown 落的就是墙钟，探测钟必须同源
    pre = await stitch_for_turn(query, user_id, course_id, db, now=t0)
    committed = False
    if pre.should_stitch and pre.evidence_topic:
        await commit_stitch_cooldown(db, user_id, course_id, pre.evidence_topic)
        committed = True
    brief = await stitch_for_turn(
        query, user_id, course_id, db, now=t0 + cooldown_hours * 3600.0
    )
    return {
        "now": t0 + cooldown_hours * 3600.0,
        "gate": _gate_fields(brief),
        "cooldown_committed": committed,
        "pre_gate": _gate_fields(pre),
    }


def _gate_fields(brief) -> dict:
    return {
        "stitch": brief.should_stitch, "reason": brief.reason,
        "p_need": round(brief.p_need, 4), "tau": round(brief.tau, 4),
        "evidence": brief.evidence_topic, "matched_topic": brief.matched_topic,
    }


async def run_check_match() -> dict:
    """预检（零 LLM）：30 条探测问句过生产 _match_topic，跑前看 matched 偏不偏。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import stitch_for_turn

    rows = []
    async with AsyncSessionLocal() as db:
        for i, c in enumerate(STITCH_CASES_LIVE):
            brief = await stitch_for_turn(
                c["query"], f"lvchk{i:02d}", COURSE_ID, db,
                now=live_case_now(c, time.time()),
            )
            actual = brief.matched_topic
            rows.append({
                "id": c["id"], "query": c["query"],
                "expected": c["matched_topic"], "actual": actual,
                "ok": actual == c["matched_topic"],
            })
    misses = [r for r in rows if not r["ok"]]
    logger.info(
        "[live] check-match：30 例命中 %d，偏离 %d%s",
        30 - len(misses), len(misses),
        "".join(f"\n  偏离 {r['id']}: 期望 {r['expected']} 实际 {r['actual']}" for r in misses),
    )
    return {"suite": "table_e_live_check_match", "cases": len(rows),
            "match_misses": len(misses), "details": rows}


async def run(case_ids: list[str] | None = None) -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from .run_stitch_table_e import ZERO_LLM_ARMS, _metrics

    await _probe_retrieval(COURSE_ID)

    cases = [c for c in STITCH_CASES_LIVE if not case_ids or c["id"] in case_ids]
    if case_ids:
        missing = set(case_ids) - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"未知 case_id: {sorted(missing)}")

    wall0 = time.time()
    base = wall0 - _CLOCK_DAYS_MAX * 86400.0  # 见 _CLOCK_DAYS_MAX 注释
    stats = {a: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for a in ZERO_LLM_ARMS}
    rows: list[dict] = []

    await init_db()
    try:
        for idx, c in enumerate(STITCH_CASES_LIVE):
            if c not in cases:
                continue
            user_id = f"lv_{idx:02d}"
            # 每例观测锚：冷却三例锚 wall0（见 _CLOCK_DAYS_MAX 注释），其余锚 base
            case_base = (
                wall0
                if ("cooldown_hours" in c and "clock_days" not in c)
                else base
            )
            logger.info("[live] === %s (%d/%d) user=%s ===", c["id"], idx + 1, len(STITCH_CASES_LIVE), user_id)
            async with AsyncSessionLocal() as db:
                await _ensure_user(db, user_id)
                await _cleanup_user(db, user_id, COURSE_ID)

            turn_logs, retry = await _expose_case(user_id, COURSE_ID, c, case_base)

            async with AsyncSessionLocal() as db:
                extract = await _read_extract(db, user_id, COURSE_ID, c)
                probe = await _probe_gate(db, user_id, COURSE_ID, c, base)

            gold = c["gold"]["stitch"]
            gate_stitch = probe["gate"]["stitch"]
            row = {
                "id": c["id"], "label": c["label"], "teacher_rule": c["teacher_rule"],
                "query": c["query"], "gold_stitch": gold,
                "gold_evidence": c["gold"].get("evidence_topic"),
                "now": probe["now"],
                "matched_topic_expected": c["matched_topic"],
                "matched_topic_actual": probe["gate"]["matched_topic"],
                "match_ok": probe["gate"]["matched_topic"] == c["matched_topic"],
                "gate": probe["gate"],
                "cooldown_committed": probe["cooldown_committed"],
                "pre_gate": probe["pre_gate"],
                "cooldown_fn": bool(
                    gold and not gate_stitch and probe["gate"]["reason"] == "cooldown"
                ),
                "expose_turns": len(turn_logs),
                "expose_logs": turn_logs,
                "rag_hit": any("rag" in (t["tools_used"] or []) for t in turn_logs),
                "consolidate_retry": retry,
                # 估算：聊天轮数 + 观测数（每段判定 ≈ 1 次判定 LLM）；mem0 persona 抽取
                # 每 session 再 +1，此处不细分——量级对就行，论文口径看 expose_turns
                "llm_calls": len(turn_logs) + extract["n_observations"],
                "judge_levels": {tid: d["actual"] for tid, d in extract["extract_detail"].items()},
                **{k: v for k, v in extract.items() if k != "extract_detail"},
                "extract_detail": extract["extract_detail"],
            }
            for arm, pred in (("never", False), ("always", True), ("gate", gate_stitch)):
                s = stats[arm]
                if gold and pred:
                    s["tp"] += 1
                elif gold and not pred:
                    s["fn"] += 1
                elif not gold and pred:
                    s["fp"] += 1
                else:
                    s["tn"] += 1
            rows.append(row)
            logger.info(
                "[live] %s gold=%s gate=%s(%s) match=%s extract_ok=%s rag_hit=%s",
                c["id"], gold, gate_stitch, probe["gate"]["reason"],
                row["match_ok"], row["extract_ok"], row["rag_hit"],
            )
    finally:
        await close_db()

    return {
        "suite": "table_e_e2e_live",
        "cases": len(rows),
        "base": base,
        "base_utc": datetime.fromtimestamp(base, timezone.utc).isoformat(),
        "arms": {a: _metrics(**stats[a]) for a in ZERO_LLM_ARMS},
        "cooldown_fn": sum(1 for r in rows if r["cooldown_fn"]),
        "match_misses": sum(1 for r in rows if not r["match_ok"]),
        "extract_full_ok": sum(1 for r in rows if r["extract_ok"]),
        "details": rows,
    }


def main() -> None:
    from . import config
    from .run_stitch_table_e import _markdown_table

    parser = argparse.ArgumentParser(
        description="表E live 全栈：真 ChatPipeline(RAG+mem0) → 巩固 → 拨钟门控（需现网栈+真实凭据）")
    parser.add_argument("--case-id", action="append", help="只跑指定 case（可多次给）")
    parser.add_argument("--check-match", action="store_true",
                        help="预检：30 探测问句过生产匹配（零 LLM，跑全量前先看 matched 偏不偏）")
    parser.add_argument("--out", type=Path, help="输出 JSON（默认 results/stitch_table_e_e2e_live.json）")
    args = parser.parse_args()

    if args.check_match:
        summary = asyncio.run(run_check_match())
        out = args.out or config.RESULTS_DIR / "stitch_table_e_live_check_match.json"
    else:
        summary = asyncio.run(run(case_ids=args.case_id))
        out = args.out or config.RESULTS_DIR / "stitch_table_e_e2e_live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[table_e_live] %s 例 -> %s", summary["cases"], out)
    if "arms" in summary:
        logger.info("\n%s", _markdown_table(summary))
    else:
        logger.info("[table_e_live] check-match 偏离 %d 例", summary["match_misses"])


if __name__ == "__main__":
    main()
