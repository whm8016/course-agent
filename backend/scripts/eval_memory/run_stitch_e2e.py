"""学情拼接门控 e2e 四臂评测：真实 consolidate 写入 → 真实 mastery 过闸。

Usage:
    python -m scripts.eval_memory.run_stitch_e2e

四臂（plan 阶段5b/5c）：
- baseline_none        不拼（沉默基线）
- baseline_unconditional 无条件拼（全拼基线）
- gate                 PRISM 单阈值（decide_stitch_prism，对照臂）
- gate_voi             VoI 三因子三档（decide_stitch，主方法）

与 scorer.score_stitch_gate（纯函数 When 决策）的区别：本脚本**走真实写入链路**——
record/seed episodes → 真实 consolidate（闭集 match_topics + mastery 写入）→ 用
knowledge_mastery 真实行过两版门控。治"种数据时直接 kp_id=topic_id 绕过写入路径"
的假绿（旧 e2e 的 _seed_case_profile 直接种对齐好的行）。

主指标：balanced accuracy（PASK 口径：TPR 与 TNR 均衡，类不平衡下不被多数类淹没）。
另报四臂的拼接率与 VoI 三档分布。零 LLM（闭集臂 embedding 离线确定性向量）。
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
    """真实写入路径：种 episodes → 走 consolidate（闭集 match_topics + mastery 落库）。

    match_topics 用确定性向量命中 gold matched_topic（mock 最近邻；mastery/risk/forget
    的演进是真实的）。mem0 双槽写入也走真实 _promote_segment——这里 mock 掉网络面
    （mem0 client），只验 DB 面。
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
    # episodes：问句即 case query；mastery 通过 mock match_topics 命中 gold 主题
    db.add(MemoryEpisode(
        user_id=user_id, course_id=course_id, session_id=f"{user_id}-s", turn_id="t1",
        user_msg=case["query"], assistant_msg="（回答略）", status="pending",
    ))
    await db.commit()

    # case 自带 mastery 状态：预先种进 knowledge_mastery（kp_id=topic_id 闭集口径，
    # 观测时间 30 天前——p_forget≈1，让 risk_used 直接起作用；近期观测的 case 由
    # get_mastery_context 路径覆盖，不在此评）
    import time as _time

    from core.db.database import KnowledgeMastery

    now = _time.time()
    for m in case["mastery"]:
        db.add(KnowledgeMastery(
            user_id=user_id, course_id=course_id, kp_id=m["topic_id"],
            label=m["topic_id"], risk=m["risk"], mastery=m.get("mastery", 0.5),
            observation_count=m.get("observation_count", 1),
            first_observed_at=now - 30 * 86400, last_observed_at=now - 30 * 86400,
        ))
    await db.commit()

    hits = [(gold, 0.9)] if gold else []
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=hits)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        await consolidation.consolidate(db, user_id, course_id)


def _balanced_accuracy(tp: int, fn: int, fp: int, tn: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return (tpr + tnr) / 2


async def run() -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from core.memory.course_topic_store import get_course_topic_map
    from core.memory.proactive import _read_mastery, decide_stitch, decide_stitch_prism

    from .stitch_cases import STITCH_CASES, prereq_predecessors

    await init_db()
    stats = {arm: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for arm in
             ("baseline_none", "baseline_unconditional", "gate", "gate_voi")}
    levels: dict[str, int] = {"silent": 0, "light": 0, "full": 0}
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
                repeated = set(c["error_repeated"])

                for arm, stitch in (
                    ("baseline_none", lambda: False),
                    ("baseline_unconditional", lambda: True),
                    ("gate", lambda: decide_stitch_prism(matched, closure, mastery_map, repeated).should_stitch),
                    ("gate_voi", lambda: decide_stitch(matched, closure, mastery_map, repeated).should_stitch),
                ):
                    pred = stitch()
                    s = stats[arm]
                    if gold_stitch and pred:
                        s["tp"] += 1
                    elif gold_stitch and not pred:
                        s["fn"] += 1
                    elif not gold_stitch and pred:
                        s["fp"] += 1
                    else:
                        s["tn"] += 1

                if matched is not None:
                    voi = decide_stitch(matched, closure, mastery_map, repeated)
                    levels[voi.level] += 1
    finally:
        await close_db()

    out = {
        arm: {**s, "balanced_accuracy": round(_balanced_accuracy(**s), 4)}
        for arm, s in stats.items()
    }
    out["voi_level_distribution"] = levels
    return out


def main() -> None:
    from . import config

    summary = asyncio.run(run())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "stitch_e2e.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm, s in summary.items():
        if isinstance(s, dict) and "balanced_accuracy" in s:
            logger.info("[stitch_e2e] %-24s bal_acc=%.3f (tp=%d fn=%d fp=%d tn=%d)",
                        arm, s["balanced_accuracy"], s["tp"], s["fn"], s["fp"], s["tn"])
    logger.info("[stitch_e2e] VoI 三档分布: %s → %s", summary["voi_level_distribution"], out)


if __name__ == "__main__":
    main()
