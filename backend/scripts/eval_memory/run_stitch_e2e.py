"""学情拼接门控 e2e 三臂评测：真实 consolidate 写入 → 真实 mastery 过闸。

Usage:
    python -m scripts.eval_memory.run_stitch_e2e

三臂（VoI 三档已于 2026-08 简化为单阈值两档，gate_voi/PRISM 对照臂随之删除）：
- baseline_none        不拼（沉默基线）
- baseline_unconditional 无条件拼（全拼基线）
- gate                 单阈值两档（decide_stitch，p_need >= τ）

与 scorer.score_stitch_gate（纯函数 When 决策）的区别：本脚本**走真实写入链路**——
record/seed episodes → 真实 consolidate（闭集 match_topics + mastery 写入）→ 用
knowledge_mastery 真实行过门控。治"种数据时直接 kp_id=topic_id 绕过写入路径"
的假绿（旧 e2e 的 _seed_case_profile 直接种对齐好的行）。

主指标：balanced accuracy（PASK 口径：TPR 与 TNR 均衡，类不平衡下不被多数类淹没）。
零 LLM（闭集臂 embedding 离线确定性向量）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)



async def _seed_case(db, course_id: str, user_id: str, case: dict) -> None:
    """真实写入路径：种 episodes → 走 consolidate（match_topics + 判定 + 观测折叠）。

    match_topics / judge_segment mock 为确定性命中（无 LLM / embedding 面），观测
    折叠（mastery_observation → knowledge_mastery.theta/risk）与读取链路全真实。
    mem0 双槽写入 mock 掉网络面（mem0 client），只验 DB 面。
    """
    from unittest.mock import AsyncMock, patch

    from core.db.database import CourseTopic, MemoryEpisode
    from core.memory import consolidation
    from .stitch_cases import COURSE_TOPICS

    # 课程主题图 + 确定性 embedding（同 run_extract_compare 口径）
    for t in COURSE_TOPICS:
        v = [((ord(c) % 97) + 1) / 97.0 for c in t["label"][:8]]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        db.add(CourseTopic(
            course_id=course_id, topic_id=t["topic_id"], label=t["label"],
            definition=t["definition"], embedding=[x / norm for x in v],
        ))
    await db.commit()

    gold = case["matched_topic"]
    # episodes：问句即 case query；mastery 通过 mock match_topics 命中 gold 主题。
    # 多轮构造观测序列：薄弱主题 poor×n / 已纠正 good×n（模拟判定历史），近 30 天前
    # 的对话（p_forget≈1，跨会话预警场景）。
    import time as _time

    from core.memory.judge import Judgement, JudgeResult

    now = _time.time()
    # 观测序列（=判定段数）：一段一次判定一条观测。薄弱主题造 4 个单轮段（4 次 poor
    # → logit -0.74 越过 WEAK_MARGIN；case 语料是「多次薄弱观测」场景），已纠正主题
    # 造 2 个 good 段。段间插入无主题轮切开（split_topic_segments 边界=主 topic 变更）。
    seg_topics: list[str | None] = []  # 每轮的主 topic（None=切段轮）
    for m in case["mastery"]:
        n_seg = 4 if m["risk"] >= 0.5 else 2
        main = m["topic_id"] if gold else "ohm"
        for _ in range(n_seg):
            seg_topics.append(None)   # 切段轮
            seg_topics.append(main)
    for i, t in enumerate(seg_topics):
        db.add(MemoryEpisode(
            user_id=user_id, course_id=course_id, session_id=f"{user_id}-s",
            turn_id=f"t{i}", user_msg=case["query"], assistant_msg="（回答略）",
            status="pending", created_at=now - 30 * 86400 + i,
        ))
    await db.commit()

    # fake match_topics：按调用序回放 seg_topics（consolidate 按 created_at 顺序逐条调用）
    call_idx = {"i": 0}

    async def _fake_match(text, cid, db_, **kw):
        t = seg_topics[call_idx["i"]] if call_idx["i"] < len(seg_topics) else None
        call_idx["i"] += 1
        return [(t, 0.9)] if t else []

    judge_result = JudgeResult(
        judgements=[
            Judgement(
                m["topic_id"],
                "poor" if m["risk"] >= 0.5 else "good",
                0.9, case["query"],
            )
            for m in case["mastery"]
        ],
        abstained=[], error_patterns=[],
        dropped_invalid_topic=0, dropped_bad_quote=0,
    )
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(side_effect=_fake_match)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=judge_result)),
    ):
        await consolidation.consolidate(db, user_id, course_id)


def _balanced_accuracy(tp: int, fn: int, fp: int, tn: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return (tpr + tnr) / 2


async def run() -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from core.memory.course_topic_store import get_course_topic_map
    from core.memory.proactive import (
        _read_error_repeated, _read_mastery, decide_stitch,
    )

    from .stitch_cases import STITCH_CASES, prereq_predecessors

    await init_db()
    stats = {arm: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for arm in
             ("baseline_none", "baseline_unconditional", "gate")}
    try:
        async with AsyncSessionLocal() as db:
            for c in STITCH_CASES:
                user_id = f"e2e_{c['id']}"
                course_id = f"course_{c['id']}"
                await _seed_case(db, course_id, user_id, c)

                gold_stitch = c["gold"]["stitch"]
                matched = c["matched_topic"]
                closure = prereq_predecessors(matched) if matched else set()
                topic_map = await get_course_topic_map(course_id, db)
                mastery_map = await _read_mastery(user_id, course_id, db, topic_map)
                # C2 复发也走真实读侧：观测日志同主题 poor 计数 ≥2（plan §3f），
                # 不直接抄 case 的 error_repeated——pos_prereq_recurrence 例的
                # 4 次 poor 观测应被 _read_error_repeated 自然数出
                repeated = await _read_error_repeated(user_id, course_id, db, topic_map)

                gate_should = decide_stitch(matched, closure, mastery_map, repeated).should_stitch
                for arm, pred in (
                    ("baseline_none", False),
                    ("baseline_unconditional", True),
                    ("gate", gate_should),
                ):
                    s = stats[arm]
                    if gold_stitch and pred:
                        s["tp"] += 1
                    elif gold_stitch and not pred:
                        s["fn"] += 1
                    elif not gold_stitch and pred:
                        s["fp"] += 1
                    else:
                        s["tn"] += 1
    finally:
        await close_db()

    return {
        arm: {**s, "balanced_accuracy": round(_balanced_accuracy(**s), 4)}
        for arm, s in stats.items()
    }


def main() -> None:
    from . import config

    summary = asyncio.run(run())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "stitch_e2e.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm, s in summary.items():
        logger.info("[stitch_e2e] %-24s bal_acc=%.3f (tp=%d fn=%d fp=%d tn=%d)",
                    arm, s["balanced_accuracy"], s["tp"], s["fn"], s["fp"], s["tn"])


if __name__ == "__main__":
    main()
