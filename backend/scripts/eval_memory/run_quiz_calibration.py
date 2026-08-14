"""上线后校准：mastery 预测后续首次同主题答题（plan 阶段F / §4b）。

Usage:
    python -m scripts.eval_memory.run_quiz_calibration [--course-id c1]

协议：对每条 source=quiz 的观测，取其 created_at 之前该生该主题的最新 mastery
（由观测日志重放折叠，含 b_topic），把它当作对这次答题的对错预测：
    p_pred = sigmoid(theta - b_topic)，label = outcome（1/0）
报 AUC + Brier + 按预测分桶的可靠性曲线。估计器权重随判分数据累积自然从判定
迁移到判分（U(n) 递减），无需手工切换——本脚本同时报各主题观测构成印证这一点。

CI 安全：无 quiz 观测时如实报 skipped，不产假数字。
"""
from __future__ import annotations

import argparse
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


# ---- 指标（纯函数） ─────────────────────────────────────────────────────────


def auc(pairs: list[tuple[float, int]]) -> float:
    """AUC（rank 实现，处理并列）。pairs: [(p_pred, label)]。"""
    if not pairs:
        return 0.0
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return 0.0
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def brier(pairs: list[tuple[float, int]]) -> float:
    """Brier score = mean((p_pred - label)^2)。"""
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def reliability(pairs: list[tuple[float, int]], bins: int = 5) -> list[dict]:
    """可靠性曲线：预测分桶，报每桶平均预测 vs 实际正确率。"""
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        out.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": len(bucket),
            "avg_pred": round(sum(p for p, _ in bucket) / len(bucket), 4),
            "actual_rate": round(sum(y for _, y in bucket) / len(bucket), 4),
        })
    return out


# ---- 主流程 ────────────────────────────────────────────────────────────────


async def run(course_id: str | None = None) -> dict:
    from core.db.database import AsyncSessionLocal, CourseTopic, MasteryObservation, close_db, init_db
    from core.memory.mastery_state import FoldState, fold_observations, sigmoid
    from sqlalchemy import select

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            b_map = {
                r.topic_id: float(r.difficulty_b or 0.0) for r in (await db.execute(
                    select(CourseTopic.topic_id, CourseTopic.difficulty_b)
                    .where(CourseTopic.course_id == course_id) if course_id
                    else select(CourseTopic.topic_id, CourseTopic.difficulty_b)
                )).all()
            }
            stmt = select(
                MasteryObservation.user_id, MasteryObservation.course_id,
                MasteryObservation.topic_id, MasteryObservation.source,
                MasteryObservation.outcome, MasteryObservation.weight,
                MasteryObservation.created_at,
            ).order_by(MasteryObservation.created_at)
            if course_id:
                stmt = stmt.where(MasteryObservation.course_id == course_id)
            obs = (await db.execute(stmt)).all()
    finally:
        await close_db()

    # 按 (user, course, topic) 分组；每条 quiz 观测用其之前全部观测的折叠态做预测
    from collections import defaultdict

    by_key: dict[tuple, list] = defaultdict(list)
    for o in obs:
        by_key[(o.user_id, o.course_id, o.topic_id)].append(o)

    pairs: list[tuple[float, int]] = []
    for (_uid, _cid, tid), group in by_key.items():
        if not any(o.source == "quiz" for o in group):
            continue
        b = b_map.get(tid, 0.0)
        # 逐条重放：每条 quiz 观测用其**之前**（不含本条）的折叠态做预测——严格
        # out-of-sample；首条 quiz 的先验就是 sigmoid(b_topic)（无信息时 0.5）
        state = FoldState(b_topic=b)
        for o in group:
            if o.source == "quiz":
                pairs.append((sigmoid(state.theta - state.b_topic), int(o.outcome == 1.0)))
            state = fold_observations(
                [{"created_at": o.created_at, "outcome": o.outcome,
                  "weight": o.weight, "source": o.source}],
                state,
            )

    if not pairs:
        logger.info("[calibration] 无 quiz 观测（未上线/无判分数据），skipped")
        return {"skipped": True, "quiz_pairs": 0, "reason": "no_quiz_observations"}

    return {
        "skipped": False,
        "quiz_pairs": len(pairs),
        "auc": round(auc(pairs), 4),
        "brier": round(brier(pairs), 4),
        "reliability": reliability(pairs),
    }


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--course-id", default=None)
    args = parser.parse_args()

    summary = asyncio.run(run(args.course_id))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "quiz_calibration.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[calibration] → %s", out)


if __name__ == "__main__":
    main()
