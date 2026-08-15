"""一次性诊断：核对生产 knowledge_mastery.kp_id 是否能与 course_topic.topic_id 对齐。

读门控 _read_mastery 要 C1（前置缺口）生效，需 mastery_map 的键（= kp_id）落在
course_topic.topic_id 口径里。本探针只 SELECT，不改数据，回答三问：
  1. kp_id 实际长什么样（kp:<hash> / LightRAG entity_id / 还是 topic_id 口径）？
  2. kp_id 是否出现在该 course 的 topic_id 集合里（即 C1 的 `tid in mastery_map` 能否成立）？
  3. 若不对齐，label 兜底（_norm_label(label)→topic_id）能救多少行？（现实里被短路，此处理论值）
"""
import asyncio
import sys
from collections import Counter

from sqlalchemy import func, select

sys.path.insert(0, ".")

from core.db.database import AsyncSessionLocal, CourseTopic, KnowledgeMastery  # noqa: E402
from core.memory.course_topic_store import _norm_label  # noqa: E402


async def main() -> None:
    async with AsyncSessionLocal() as db:
        total = (
            await db.execute(select(func.count()).select_from(KnowledgeMastery))
        ).scalar()
        print(f"knowledge_mastery 总行数: {total}")
        if not total:
            print("(空表 — 没有真实 mastery 数据可看)")
            return

        rows = (
            await db.execute(
                select(
                    KnowledgeMastery.kp_id,
                    KnowledgeMastery.label,
                    KnowledgeMastery.course_id,
                    KnowledgeMastery.risk,
                ).order_by(KnowledgeMastery.course_id)
            )
        ).all()

        # --- Q1: kp_id 形态分布 ---
        def shape(kp: str) -> str:
            if kp.startswith("kp:"):
                return "kp:<sha1hash>"  # _kp_id_from 兜底路径
            return "other(LightRAG entity_id?)"  # 目录匹配路径

        dist = Counter(shape(r[0]) for r in rows)
        print(f"\n[Q1] kp_id 形态分布: {dict(dist)}")
        print("样本（kp_id | label | course_id | risk）:")
        for r in rows[:12]:
            print(f"  {r[0]!r:40} | {str(r[1])[:18]:18} | {r[2]} | {r[3]}")

        # --- 按 course 汇总，逐课检查对齐 ---
        topics_by_course: dict[str, set[str]] = {}
        for trow in (
            await db.execute(
                select(CourseTopic.course_id, CourseTopic.topic_id, CourseTopic.label)
            )
        ).all():
            topics_by_course.setdefault(trow[0], set()).add(trow[1])

        print(f"\ncourse_topic 覆盖课程数: {len(topics_by_course)}")
        for cid, tids in topics_by_course.items():
            sample = list(tids)[:5]
            print(f"  course={cid} topic 数={len(tids)} 样本topic_id={sample}")

        # --- Q2: kp_id 是否落在该 course 的 topic_id 集 ---
        aligned = 0
        misaligned = 0
        # --- Q3: label 兜底能否救 ---
        label_rescued = 0
        no_course_topics = 0
        for kp_id, label, course_id, _risk in rows:
            tids = topics_by_course.get(course_id)
            if not tids:
                no_course_topics += 1
                continue
            if kp_id in tids:
                aligned += 1
            else:
                misaligned += 1
                # label 兜底理论值（生产里被 r.kp_id or ... 短路，永不执行）
                if _norm_label(label or "") and any(
                    _norm_label(label or "") == _norm_label(t) for t in tids
                ):
                    label_rescued += 1

        print("\n========== 对齐核验 ==========")
        print(f"kp_id 直接 == 某 topic_id （C1 现实可达） : {aligned}")
        print(f"kp_id 不在任何 topic_id 集（C1 现实不可达）: {misaligned}")
        print(f"  其中 label 兜底理论可救                   : {label_rescued}")
        print(f"  该 course 无 course_topic（连兜底都没有）  : {no_course_topics}")

        if misaligned and not aligned:
            print("\n>> 结论：生产里 C1（前置缺口）的 `tid in mastery_map` 恒为假，分支死。")


if __name__ == "__main__":
    asyncio.run(main())
