"""一次性诊断：真实跑一遍 consolidate 的抽取写入口径，看 kp_id 长什么样。

对 mycourse（真实 34 主题，topic_id=_norm_label(label)[:32] 中文）跑
graph_memory.extract_knowledge（consolidate job 用的同一路径），再用
mastery._kp_id_from 算出会落库的 kp_id，与该课 topic_id 比对。

只读 + LLM 抽取，不 append_mastery、不写库。
"""
import asyncio
import sys

sys.path.insert(0, ".")

from settings import get_settings  # noqa: E402,F401  (触发 .env 加载)

from core.memory import graph_memory  # noqa: E402
from core.memory.course_topic_store import _norm_label  # noqa: E402
from core.memory.mastery import _kp_id_from  # noqa: E402
from sqlalchemy import select  # noqa: E402
from core.db.database import AsyncSessionLocal, CourseTopic  # noqa: E402

COURSE = "mycourse"
# 问一个该课真实存在的主题（戴维南定理）
Q = "老师，戴维南定理我怎么都算不对，等效电阻到底怎么求？开路电压测了还是错。"
A = (
    "戴维南定理：把线性二端网络等效成一个电压源 U_oc 串联一个等效电阻 R_eq。"
    "R_eq 是把独立源置零后从端口看进去的等效电阻。你开路电压测错可能是仪表内阻影响。"
)


async def main() -> None:
    from core.rag.lightrag.llm_adapter import is_lightrag_available

    print(f"LightRAG 可用: {is_lightrag_available()}  (True=目录匹配路径, False=开放提取兜底)")
    print(f"LLM text_model: {get_settings().llm.text_model}")

    # 该课真实 topic_id 集
    async with AsyncSessionLocal() as db:
        tids = {
            r[0]
            for r in (
                await db.execute(
                    select(CourseTopic.topic_id).where(CourseTopic.course_id == COURSE)
                )
            ).all()
        }
    print(f"\n{COURSE} 真实 topic_id 集（前6）: {sorted(tids)[:6]}")

    print("\n=== 运行 extract_knowledge（consolidate 同路径）===")
    extracted = await graph_memory.extract_knowledge(COURSE, Q, A)
    kps = (extracted or {}).get("knowledge_points") or []
    print(f"抽出 knowledge_points: {len(kps)} 个")
    if not kps:
        print("(LLM 未抽出知识点或调用失败——见上方日志)")
        return

    print("\n=== 每个知识点会落库的 kp_id vs 该课 topic_id ===")
    print(f"{'label':22} | {'会落库的 kp_id':28} | {'理想 topic_id(=norm_label[:32])':30} | 落在topic集?")
    any_aligned = False
    for kp in kps:
        label = str(kp.get("label") or "").strip()
        kp_id = _kp_id_from(kp)
        want = _norm_label(label)[:32]
        hit = kp_id in tids
        any_aligned = any_aligned or hit
        eid = kp.get("entity_id")
        src = "catalog(entity_id)" if eid else "fallback→kp:<hash>"
        print(f"{label[:22]:22} | {kp_id[:28]:28} | {want[:30]:30} | {hit}  [{src}]")

    print(
        "\n>> 结论：consolidate 写出的 kp_id 与 topic_id 对齐？ "
        f"{'是（意外）' if any_aligned else '否 —— C1 在真实 consolidate 数据下不可达'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
