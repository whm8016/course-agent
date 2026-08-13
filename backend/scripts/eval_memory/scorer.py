"""L3 记忆评测 scorer——照 LongMemEval 的 knowledge updates / abstention 维度，
对我们 mastery 层做程序化判分（无需 LLM，可入 CI）。

三维：
- knowledge_update：掌握度能否随观测正确演进（连续改善 → risk 降；连续退步 → risk 升）
- abstention：无数据 / 仅低置信时，get_mastery_context 不得编造薄弱点（返回空 / 不列低风险）
- decay：旧观测按 last_observed_at 软衰减——排序靠后但不物理删除（反复性错误证据留存）

每个 scorer 在独立 user_id 上跑（eval DB 内隔离），返回 (passed, total, details)。
"""
from __future__ import annotations

import time
from typing import Any


async def _read_mastery(db, user_id: str, course_id: str, kp_id: str) -> Any:
    from sqlalchemy import select

    from core.db.database import KnowledgeMastery

    return (
        (
            await db.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == user_id,
                    KnowledgeMastery.course_id == course_id,
                    KnowledgeMastery.kp_id == kp_id,
                )
            )
        )
        .scalars()
        .first()
    )


async def score_knowledge_update(db) -> tuple[int, int, list[str]]:
    """掌握度随观测正确演进：改善场景 + 退步场景。"""
    from core.memory import mastery

    passed = 0
    total = 0
    details: list[str] = []

    # 场景 1：先薄弱(risk↑)再连续改善(risk↓) → 最终 risk 应低于峰值
    total += 1
    await mastery.append_mastery(
        db, "eval_ku", "c1",
        [{"label": "导数", "entity_id": "e1", "mastery_delta": 0.0, "risk_delta": 0.3}], "ep1",
    )
    # 注意：须立即快照标量——ORM 行对象在 identity map 中会被后续 append原地突变，
    # 持引用到比较时读到的已是终值（mistake 出过：peak/after 都读成 0.55）。
    row_peak = await _read_mastery(db, "eval_ku", "c1", "e1")
    risk_peak = row_peak.risk if row_peak else None
    await mastery.append_mastery(
        db, "eval_ku", "c1",
        [{"label": "导数", "entity_id": "e1", "mastery_delta": 0.2, "risk_delta": -0.25}], "ep2",
    )
    row_after = await _read_mastery(db, "eval_ku", "c1", "e1")
    risk_after = row_after.risk if row_after else None
    if risk_after is not None and risk_peak is not None and risk_after < risk_peak:
        passed += 1
    else:
        details.append(f"ku_improve: risk 未下降 peak={risk_peak} after={risk_after}")

    # 场景 2：先扎实(risk↓)再连续出错(risk↑) → 最终 risk 应高于谷值
    total += 1
    await mastery.append_mastery(
        db, "eval_ku2", "c1",
        [{"label": "积分", "entity_id": "e2", "mastery_delta": 0.2, "risk_delta": -0.3}], "ep1",
    )
    row_trough = await _read_mastery(db, "eval_ku2", "c1", "e2")
    risk_trough = row_trough.risk if row_trough else None
    await mastery.append_mastery(
        db, "eval_ku2", "c1",
        [{"label": "积分", "entity_id": "e2", "mastery_delta": -0.2, "risk_delta": 0.3}], "ep2",
    )
    row_final = await _read_mastery(db, "eval_ku2", "c1", "e2")
    risk_final = row_final.risk if row_final else None
    if risk_final is not None and risk_trough is not None and risk_final > risk_trough:
        passed += 1
    else:
        details.append(f"ku_regress: risk 未上升 trough={risk_trough} final={risk_final}")

    return passed, total, details


async def score_abstention(db) -> tuple[int, int, list[str]]:
    """无数据 / 仅低置信时不得编造薄弱点。"""
    from core.memory import mastery

    passed = 0
    total = 0
    details: list[str] = []

    # 场景 1：无任何数据 → get_mastery_context 必须空串（不编造）
    total += 1
    ctx_empty = await mastery.get_mastery_context(db, "eval_ab_empty", "c1")
    if ctx_empty == "":
        passed += 1
    else:
        details.append(f"ab_empty: 无数据却返回了 {ctx_empty!r}")

    # 场景 2：只有低风险(risk<0.5)知识点 → 不得列为薄弱点
    total += 1
    await mastery.append_mastery(
        db, "eval_ab_low", "c1",
        [{"label": "加减法", "entity_id": "e3", "mastery_delta": 0.4, "risk_delta": -0.4}], "ep1",
    )
    ctx_low = await mastery.get_mastery_context(db, "eval_ab_low", "c1")
    if "加减法" not in ctx_low:
        passed += 1
    else:
        details.append("ab_low: 低风险知识点被误列为薄弱（应 abstain）")

    return passed, total, details


async def score_decay(db) -> tuple[int, int, list[str]]:
    """旧高险观测软衰减：近期较低险应排在旧的更高险之前（衰减后），但旧的不被删除。"""
    from core.db.database import KnowledgeMastery
    from core.memory import mastery

    passed = 0
    total = 0
    details: list[str] = []

    now = time.time()
    # 旧的高风险（90 天前）：raw risk 0.9，衰减后 effective ≈ 0.9·exp(-0.9) ≈ 0.37
    # 近期中风险（现在）：raw risk 0.6，effective 0.6 → 排更前
    db.add(KnowledgeMastery(
        user_id="eval_decay", course_id="c1", kp_id="old", label="旧难点",
        mastery=0.2, risk=0.9, observation_count=1,
        first_observed_at=now - 90 * 86400, last_observed_at=now - 90 * 86400,
        evidence_episode_ids=[],
    ))
    db.add(KnowledgeMastery(
        user_id="eval_decay", course_id="c1", kp_id="recent", label="近难点",
        mastery=0.4, risk=0.6, observation_count=1,
        first_observed_at=now, last_observed_at=now, evidence_episode_ids=[],
    ))
    await db.commit()

    total += 1
    ctx = await mastery.get_mastery_context(db, "eval_decay", "c1")
    # 近难点（effective 0.6）应排在旧难点（effective 0.37）之前
    if "近难点" in ctx and "旧难点" in ctx and ctx.index("近难点") < ctx.index("旧难点"):
        passed += 1
    else:
        details.append(f"decay: 衰减排序错误 ctx={ctx!r}")

    return passed, total, details


async def score_stitch_gate(db) -> tuple[int, int, list[str]]:
    """拼接门控 When 决策正确性：decide_stitch 对 stitch_cases 全部 case 的 should 匹配 gold。

    零 LLM（decide_stitch 纯函数 + stitch_cases 内存数据，无需 DB，但签名与其它 scorer 一致）。
    正例（前置缺口/复发）该拼、负例（unknown/无关/已纠正/全低险/阶段错位）不该拼。
    mastery 用 risk 作 eff_risk（case 为近期观测，衰减近似无）。
    """
    from core.memory.proactive import decide_stitch

    from .stitch_cases import STITCH_CASES, prereq_predecessors

    passed = 0
    total = 0
    details: list[str] = []
    for c in STITCH_CASES:
        total += 1
        matched = c["matched_topic"]
        closure = prereq_predecessors(matched) if matched else set()
        mastery = {
            m["topic_id"]: {"risk": m["risk"], "eff_risk": m["risk"]}
            for m in c["mastery"]
        }
        repeated = set(c["error_repeated"])
        brief = decide_stitch(matched, closure, mastery, repeated)
        if brief.should_stitch == c["gold"]["stitch"]:
            passed += 1
        else:
            details.append(f"{c['id']}: should={brief.should_stitch} gold={c['gold']['stitch']}")
    return passed, total, details


SCORERS = {
    "knowledge_update": score_knowledge_update,
    "abstention": score_abstention,
    "decay": score_decay,
    "stitch_gate": score_stitch_gate,
}
