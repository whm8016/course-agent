"""mycourse 主题抽取 live 测评：重建后统计图质量 + 模拟学生问句最近邻命中。

先跑 build_course_topics --from-chunks --force，再跑本脚本。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

JSON_PATH = Path("data/course_topics/course_mycourse_topics.json")

# 课上学生会问的话（覆盖大纲粗点、实验细点、图注碎片、跨章）
STUDENT_QUERIES = [
    "示波器怎么触发？波形跑了怎么调？",
    "示波器探头怎么接？要不要补偿？",
    "戴维南定理怎么用？怎么求等效电压和等效电阻？",
    "等效内阻怎么测量？开路短路法具体步骤是什么？",
    "叠加定理怎么验证？",
    "RC低通滤波器频率特性怎么测？",
    "三端变阻器怎么接？",
    "通道设置菜单怎么调？",  # 旧图注碎片，期望不要打到菜单垃圾点
    "滤波器实验怎么做？",
]


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def graph_report(data: dict) -> dict:
    topics = data["topics"]
    edges = data.get("edges") or []
    stats = data.get("stats") or {}
    secs = Counter(t.get("source_section") for t in topics)
    unknown = secs.get("未知章节", 0)
    outline = [t for t in topics if "大纲" in str(t.get("source_section") or "")]
    labels = [t["label"] for t in topics]
    return {
        "n_topics": len(topics),
        "n_edges": len(edges),
        "unknown_section_topics": unknown,
        "outline_topics": [{"label": t["label"], "section": t["source_section"]} for t in outline],
        "filename_section_topics": [
            {"label": t["label"], "section": t["source_section"]}
            for t in topics
            if str(t.get("source_section") or "").endswith((".docx", ".pdf", ".pptx", ".txt"))
        ],
        "sections": dict(secs.most_common()),
        "stats": stats,
        "failed_sections": data.get("failed_sections") or [],
        "labels": labels,
    }


def near_dup_pairs(topics: list[dict], floor=0.70, ceiling=0.88) -> list[dict]:
    """重建后仍落在中间带的点对 = LLM 归并没吃掉的近似重复。"""
    out = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            a, b = topics[i], topics[j]
            sim = _cosine(a.get("embedding") or [], b.get("embedding") or [])
            if floor <= sim < ceiling:
                out.append({
                    "sim": round(sim, 3),
                    "a": a["label"], "a_sec": a.get("source_section"),
                    "b": b["label"], "b_sec": b.get("source_section"),
                })
    out.sort(key=lambda x: -x["sim"])
    return out[:20]


async def match_queries(topics: list[dict]) -> list[dict]:
    from core.rag.llamaindex.pg_store import get_embed_model

    embed = get_embed_model()
    rows = []
    for q in STUDENT_QUERIES:
        qe = await embed._aget_text_embedding(q)
        ranked = sorted(
            ((_cosine(qe, t.get("embedding") or []), t) for t in topics),
            key=lambda x: -x[0],
        )
        top = ranked[:3]
        rows.append({
            "query": q,
            "top1": {"label": top[0][1]["label"], "sim": round(top[0][0], 3),
                     "section": top[0][1].get("source_section")} if top else None,
            "top2": {"label": top[1][1]["label"], "sim": round(top[1][0], 3),
                     "section": top[1][1].get("source_section")} if len(top) > 1 else None,
            "gap": round(top[0][0] - top[1][0], 3) if len(top) > 1 else None,
        })
    return rows


def verdict(report: dict, dups: list, matches: list) -> list[str]:
    notes = []
    if report["n_topics"] >= 123:
        notes.append("主题数没降（旧图 123），过滤或归并可能没生效")
    else:
        notes.append(f"主题数 {report['n_topics']}（旧 123）")
    if report["unknown_section_topics"] == 0:
        notes.append("未知章节 = 0")
    else:
        notes.append(f"仍有 {report['unknown_section_topics']} 个未知章节主题")
    notes.append(
        f"LLM 归并 {report['stats'].get('topics_merged_llm')} 个；"
        f"向量归并 {report['stats'].get('topics_merged_embedding')} 个"
    )
    tight = [m for m in matches if m.get("gap") is not None and m["gap"] < 0.03]
    if tight:
        notes.append(f"{len(tight)} 条问句 top1/top2 分差 <0.03，仍可能打到两个相似点")
    else:
        notes.append("模拟问句 top1 与 top2 分差都 ≥0.03")
    notes.append(f"重建后中间带近对 {len(dups)} 对（列出前 20）")
    return notes


async def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"没有 {JSON_PATH}，先跑 from-chunks --force")
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    report = graph_report(data)
    dups = near_dup_pairs(data["topics"])
    matches = await match_queries(data["topics"])
    out = {
        "graph": {k: v for k, v in report.items() if k != "labels"},
        "labels": report["labels"],
        "near_dup_pairs": dups,
        "student_query_match": matches,
        "verdict": verdict(report, dups, matches),
    }
    dest = Path("scripts/eval_memory/results/topic_extract_live_eval.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "n_topics": report["n_topics"],
                      "unknown": report["unknown_section_topics"],
                      "stats": report["stats"],
                      "failed": report["failed_sections"]}, ensure_ascii=False, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
