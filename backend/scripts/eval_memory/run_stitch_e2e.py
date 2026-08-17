"""表E e2e 评测：生产入口 stitch_for_turn 全链（真实匹配 + 库内时间 + 复发自然长出）。

Usage（backend 目录，需 EMBEDDING__API_KEY——问句要真实向量化）:
    python -m scripts.eval_memory.run_stitch_e2e

与离线决策层（run_stitch_table_e：decide_stitch + 档案给定 matched_topic）的区别
是评测去泄漏三件套：
- **入口是生产读侧** ``core.memory.proactive.stitch_for_turn``：问句只有一句中文，
  matched_topic 由 ``_match_topic`` 拿 course_topic.embedding 真最近邻算出（旧 e2e
  把 case 的 matched_topic 塞回 decide_stitch，匹配这步是假的——已删）；
- **构造侧直写生产读的字段**：mastery_observation.created_at（30 天复发窗）、
  knowledge_mastery.last_observed_at（遗忘）/ last_stitched_at（冷却）/ theta
  （fold_observations 真实折出，b 用快照真难度，列集对齐 consolidation 生产 INSERT）；
- **发问侧只传 query + 冻结 now**（case_now）：遗忘/复发/冷却全部由管道自己比
  「现在」算出，出题人不再预折 p_forget、预写 error_repeated。

金标=教师规则句子（种进去的世界）；门控因匹配错/冷却/真难度偏离 = 评测的信息量：
match_ok / cooldown_fn 单独记列，冷却 FN 留在混淆矩阵不剔除。

图源 graph_mycourse_v3_emb.json（= v2 图 + label 重算真实向量，见 export_graph_snapshot
的口径说明）：193 主题全量种进内存 sqlite，全 case 共用一个 course、隔离靠 user_id
（同生产读域）。复发口径自检：落库后 _read_error_repeated 长出的集合必须与 case
声明的 error_repeated 一致，不等先炸——那是案例数据 bug，不是评测结果。

零 LLM（决策臂不调模型；embedding 探针 + 逐例问句向量化走真实 embedding API）。
离线接线自检（CI）：tests/test_stitch_e2e_wiring.py 打桩 embed 单例跑同一条 run()。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# 档案构造单一真相源（与离线 build_mastery_map 同一份）：env 预置之后再导入，
# stitch_cases_table_e 模块级无 core 依赖，此处安全
from .stitch_cases_table_e import (  # noqa: E402
    STITCH_CASES, case_now, entry_last_stitched_at, entry_observations, fold_entry,
)

_SNAPSHOT_V3 = Path(__file__).resolve().parent / "graph_mycourse_v3_emb.json"


def load_graph_embedded() -> dict:
    """v3 快照（e2e 图源）：{course_id, topics:[{topic_id,label,difficulty_b,
    safety_weight,embedding}], edges:[{src,dst,relation}]}。"""
    if not _SNAPSHOT_V3.exists():
        raise SystemExit(
            f"缺 {_SNAPSHOT_V3.name}——先跑 "
            "python -m scripts.eval_memory.export_graph_snapshot（需 EMBEDDING__API_KEY）"
        )
    return json.loads(_SNAPSHOT_V3.read_text(encoding="utf-8"))


async def _probe_embed() -> None:
    """凭据闸（表A 的 fail-fast 口径，查 embedding 侧非 rerank）：真调一次向量化。
    缺 key 时 30 例会静默全 unknown（_match_topic fail-safe），必须先拦。测试打桩
    embed 单例时走桩，天然通过。"""
    from core.rag.llamaindex.pg_store import get_embed_model

    try:
        model = get_embed_model()
        await model._aget_query_embedding("测试")
    except Exception as exc:
        raise SystemExit(
            f"embedding 探针失败（缺 EMBEDDING__API_KEY？e2e 的问句向量化必须真实）：{exc}"
        ) from exc


async def _ensure_user(db, user_id: str) -> None:
    """FK 需要（knowledge_mastery / mastery_observation 都挂 users.id）。"""
    from core.db.database import User
    from sqlalchemy import select

    row = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if row is None:
        db.add(User(id=user_id, username=user_id, password_hash="seed", role="student"))
        await db.commit()


async def _seed_graph(db, snapshot: dict) -> None:
    """全 case 共用一个 course，图种一次（幂等：先清同 course 旧行）。"""
    from core.db.database import CourseTopic, CourseTopicEdge
    from sqlalchemy import delete

    course_id = snapshot["course_id"]
    await db.execute(delete(CourseTopicEdge).where(CourseTopicEdge.course_id == course_id))
    await db.execute(delete(CourseTopic).where(CourseTopic.course_id == course_id))
    for t in snapshot["topics"]:
        db.add(CourseTopic(
            course_id=course_id, topic_id=t["topic_id"], label=t["label"],
            embedding=t["embedding"],
            safety_weight=t.get("safety_weight", 1.0),
            difficulty_b=t.get("difficulty_b", 0.0),
        ))
    for e in snapshot["edges"]:
        db.add(CourseTopicEdge(
            course_id=course_id, src_topic_id=e["src"], dst_topic_id=e["dst"],
            relation=e["relation"],
        ))
    await db.commit()


async def _seed_case(
    db, course_id: str, user_id: str, case: dict, b_map: dict[str, float]
) -> None:
    """种生产读侧消费的痕迹（幂等）：观测日志 + 掌握度行（真实折叠）。

    列集对齐 consolidation 生产 INSERT（mastery/risk round 4、theta round 6、
    n_obs=observation_count）；theta 从快照真难度 b 起折（生产同 regime）。旧 e2e
    的 episodes→consolidate（mock match_topics/judge）已删——判定面不是门控的
    评测对象，观测直落即「真学生留下的痕迹」。
    """
    from core.db.database import KnowledgeMastery, MasteryObservation
    from core.memory.mastery_state import W_JUDGE
    from sqlalchemy import delete

    await db.execute(delete(MasteryObservation).where(
        MasteryObservation.user_id == user_id,
        MasteryObservation.course_id == course_id,
    ))
    await db.execute(delete(KnowledgeMastery).where(
        KnowledgeMastery.user_id == user_id,
        KnowledgeMastery.course_id == course_id,
    ))
    await _ensure_user(db, user_id)

    for m in case["mastery"]:
        obs = entry_observations(case, m)
        for o in obs:
            db.add(MasteryObservation(
                user_id=user_id, course_id=course_id, topic_id=m["topic_id"],
                source="llm_judge", outcome=o["outcome"], weight=W_JUDGE,
                confidence=0.9, evidence_quote=case["query"],
                created_at=o["created_at"],
            ))
        state = fold_entry(case, m, b_topic=b_map.get(m["topic_id"], 0.0))
        db.add(KnowledgeMastery(
            user_id=user_id, course_id=course_id, kp_id=m["topic_id"],
            label=m.get("label", m["topic_id"]),
            mastery=round(state.mastery, 4), risk=round(state.risk, 4),
            observation_count=state.n_obs,
            first_observed_at=min(o["created_at"] for o in obs),
            last_observed_at=max(o["created_at"] for o in obs),
            theta=round(state.theta, 6), n_obs=state.n_obs,
            strength=5.0,  # 不模拟 teacher_affirm 的 +1（案例不建这个场景）
            last_stitched_at=entry_last_stitched_at(case, m),
        ))
    await db.commit()


async def _assert_emergent_error_repeated(
    db, course_id: str, user_id: str, topic_map: dict[str, str], case: dict
) -> None:
    """复发口径自检（评分前）：读侧长出的集合 == case 声明。不等 = 案例数据与
    观测落点打架（harness 自检，不是评测断言），先修案例再谈分数。"""
    from core.memory.proactive import _read_error_repeated

    emergent = await _read_error_repeated(
        user_id, course_id, db, topic_map, now=case_now(case))
    declared = set(case["error_repeated"])
    if emergent != declared:
        raise RuntimeError(
            f"[{case['id']}] 复发自检不符：落库长出 {sorted(emergent)} ≠ 声明 {sorted(declared)}"
            "——观测落点（days_since_last/间距）与 error_repeated 口径打架，先修案例"
        )


async def run() -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db
    from core.memory.course_topic_store import get_course_topic_map
    from core.memory.proactive import stitch_for_turn

    from .run_stitch_table_e import ZERO_LLM_ARMS, _metrics

    await _probe_embed()

    snapshot = load_graph_embedded()
    course_id = snapshot["course_id"]
    b_map = {t["topic_id"]: t.get("difficulty_b", 0.0) for t in snapshot["topics"]}
    stats = {a: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for a in ZERO_LLM_ARMS}
    rows: list[dict] = []

    await init_db()
    try:
        # :memory: 全程一个 AsyncSession：aiosqlite :memory: 默认 StaticPool 单连接
        # （跨会话虽共享），单会话纪律免依赖池行为——换引擎/池参数不破
        async with AsyncSessionLocal() as db:
            await _seed_graph(db, snapshot)
            topic_map = await get_course_topic_map(course_id, db)

            for i, c in enumerate(STITCH_CASES):
                user_id = f"e2e_{i:02d}"  # users.id String(32)；用 case id 拼会超长
                await _seed_case(db, course_id, user_id, c, b_map)
                await _assert_emergent_error_repeated(db, course_id, user_id, topic_map, c)

                # 发问侧只做一件事：丢问句 + 冻结 now。matched/error_repeated/p_forget
                # 全部由生产读侧自己算——这是与旧 e2e（塞回 matched_topic）的本质区别
                brief = await stitch_for_turn(
                    c["query"], user_id, course_id, db, now=case_now(c))

                gold = c["gold"]["stitch"]
                row = {
                    "id": c["id"], "label": c["label"], "teacher_rule": c["teacher_rule"],
                    "query": c["query"], "gold_stitch": gold,
                    "gold_evidence": c["gold"].get("evidence_topic"),
                    "now": case_now(c),
                    "matched_topic_expected": c["matched_topic"],
                    "matched_topic_actual": brief.matched_topic,
                    "match_ok": brief.matched_topic == c["matched_topic"],
                    "gate": {
                        "stitch": brief.should_stitch, "reason": brief.reason,
                        "p_need": round(brief.p_need, 4), "tau": round(brief.tau, 4),
                        "evidence": brief.evidence_topic,
                    },
                    # 冷却 FN 单独记列（设计争议口径），但保留在混淆矩阵里
                    "cooldown_fn": bool(
                        gold and not brief.should_stitch and brief.reason == "cooldown"
                    ),
                }
                for arm, pred in (
                    ("never", False), ("always", True), ("gate", brief.should_stitch),
                ):
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
    finally:
        await close_db()

    return {
        "suite": "table_e_e2e",
        "cases": len(STITCH_CASES),
        "arms": {a: {**_metrics(**stats[a]), "llm_calls_per_case": 0} for a in ZERO_LLM_ARMS},
        "cooldown_fn": sum(1 for r in rows if r["cooldown_fn"]),
        "match_misses": sum(1 for r in rows if not r["match_ok"]),
        "details": rows,
    }


def main() -> None:
    from . import config
    from .run_stitch_table_e import _markdown_table

    parser = argparse.ArgumentParser(
        description="表E e2e：stitch_for_turn 全链评测（需 embedding 凭据）")
    parser.add_argument("--out", type=Path, help="输出 JSON（默认 results/stitch_table_e_e2e.json）")
    args = parser.parse_args()

    summary = asyncio.run(run())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or config.RESULTS_DIR / "stitch_table_e_e2e.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "[table_e_e2e] %d 例 cooldown_fn=%d match_misses=%d -> %s\n%s",
        summary["cases"], summary["cooldown_fn"], summary["match_misses"], out,
        _markdown_table(summary),
    )
    for r in summary["details"]:
        if not r["match_ok"]:
            logger.info(
                "[table_e_e2e] 匹配偏离 %s: 期望 %s 实际 %s",
                r["id"], r["matched_topic_expected"], r["matched_topic_actual"],
            )
        if r["gate"]["stitch"] != r["gold_stitch"]:
            logger.info(
                "[table_e_e2e] 决策偏离 %s: gold=%s gate=%s reason=%s p_need=%s",
                r["id"], r["gold_stitch"], r["gate"]["stitch"],
                r["gate"]["reason"], r["gate"]["p_need"],
            )


if __name__ == "__main__":
    main()
