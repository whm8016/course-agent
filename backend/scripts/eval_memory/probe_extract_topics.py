r"""诊断探针：验证「从讲义切块同抽主题+课文先修边」这条 LLM 调用链是否稳定、产物是否像样。

与 ``core.memory.course_topic_builder.build_topic_graph`` 的区别——本脚本**只测不灌**：
  - 不写 course_topic 表、不导出 JSON、不算 embedding、不做删环/闭包
  - 逐块调用**真代码路径** ``extract_block_graph``（含 gleaning 补抽、闭集端点校验、
    引句子串校验——不再自拼 prompt/自剥围栏），逐块打印主题数/边数/闭集弃边数，
    失败不中断
  - 块序与生产一致：已抽清单随块累积，后面块的边可引用前面块的点（跨章边来源）

跑通它回答四个问题：
  1. LLM 调用稳定率（N 块成功几块）
  2. 挂的是哪些块（返回 None = 两试皆败，看 WARNING 日志定位熔断/JSON）
  3. 主题质量（看末尾 label 清单，人工核是否像「可考核知识点」）
  4. 边质量（看末尾边清单：端点是否成立、引句是否真是原文、strength 是否合理）

用法（在配了真实 LLM 的环境，如 docker exec backend）：
    python -m scripts.eval_memory.probe_extract_topics --chunks <adapted.json>
    python -m scripts.eval_memory.probe_extract_topics --chunks <adapted.json> --max-chars 3000 --gleaning 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

# 仿 run.py：连真实库前先兜底环境变量（脚本可独立跑）
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

# 复用真代码的块级抽取函数与分块逻辑，保证测的就是生产路径（抽取逻辑已下沉 core）
from core.memory.course_topic_builder import _pack_blocks, extract_block_graph  # noqa: E402
from core.memory.course_topic_store import _norm_label  # noqa: E402

# WARNING 级：压掉 INFO 噪音，但保留闭集弃边/熔断器的 WARNING（看质量信号）
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def probe(chunks_path: str, max_chars: int, model: str | None = None, gleaning: int | None = None) -> None:
    from core.db.database import init_db
    from core.llm.llm import client as llm
    from settings import get_settings

    await init_db()
    s = get_settings()
    model = model or s.course_topic.extract_model or s.llm.text_model
    # gleaning 默认跟 settings 走（与 model 同模式）：probe 测的就是生产形态，
    # 部署侧改默认这里不该漂移
    gleaning = s.course_topic.gleaning if gleaning is None else gleaning

    raw = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw

    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    total_blocks = sum(len(_pack_blocks(g, max_chars)) for g in by_section.values())
    print(f"\nmodel={model}  章节={len(by_section)}  切块后总块数={total_blocks}  "
          f"max_chars={max_chars}  gleaning={gleaning}")
    print("=" * 96)
    print(f"{'#':<4}{'章节':<22}{'字符':>6}{'主题':>5}{'新边':>5}{'弃边':>5}{'带引句':>7}  备注")
    print("-" * 96)

    all_labels: list[str] = []
    all_edges: list[dict] = []
    known: dict[str, str] = {}  # 与生产同构：归一 label → label，随块累积（跨章边闭集）
    n_ok = n_fail = n_topics = n_edges = n_dropped = 0
    i = 0
    for sec, group in by_section.items():
        for block in _pack_blocks(group, max_chars):
            i += 1
            text = "\n".join(ck["text"] for ck in block)
            res = await extract_block_graph(
                "probe", sec, text, known, llm, model, gleaning=gleaning,
            )
            if res is None:  # 两试皆败（含熔断），继续下一块
                n_fail += 1
                print(f"{i:<4}{sec[:20]:<22}{len(text):>6}{'':>5}{'':>5}{'':>5}{'':>7}  FAIL（两试皆败，看上方 WARNING）")
                continue
            topics, edges, dropped = res
            with_ev = sum(1 for e in edges if e["evidence"])
            n_ok += 1
            n_topics += len(topics)
            n_edges += len(edges)
            n_dropped += dropped
            note = "⚠闭集弃边多（端点写法漂移？）" if dropped > len(edges) else ""
            print(f"{i:<4}{sec[:20]:<22}{len(text):>6}{len(topics):>5}{len(edges):>5}"
                  f"{dropped:>5}{with_ev:>7}  {note}")
            for t in topics:
                all_labels.append(t["label"])
                known[_norm_label(t["label"])] = t["label"]
            all_edges.extend(edges)

    print("=" * 96)
    print(f"汇总：{n_ok} 块成功 / {n_fail} 块失败，抽到 {n_topics} 个主题（去重前）、"
          f"{n_edges} 条边（闭集弃 {n_dropped} 条、带原文引句 {sum(1 for e in all_edges if e['evidence'])} 条）")
    print("\n抽到的全部主题 label（人工核：是「可考核知识点」还是「小标题/实验器材」？）：")
    for j, lb in enumerate(all_labels, 1):
        print(f"  {j:>3}. {lb}")
    print("\n课文锚定边清单（人工核：端点是否成立 / 引句是否原文 / 方向对不对）：")
    for j, e in enumerate(all_edges, 1):
        ev = f"「{e['evidence']}」" if e["evidence"] else "（无引句）"
        print(f"  {j:>3}. {e['src_label']} → {e['dst_label']}  [{e['confidence']:.2f}]  {ev}")


def main() -> None:
    p = argparse.ArgumentParser(description="诊断探针：测「切块同抽主题+课文边」LLM 调用链（不灌库）")
    p.add_argument("--chunks", required=True, help="适配后的切块审计 JSON 路径")
    p.add_argument("--max-chars", type=int, default=5000, help="单次调用喂给 LLM 的章节字符上限")
    p.add_argument("--model", help="覆盖 extract_model（默认 deepseek-v4-flash）")
    p.add_argument("--gleaning", type=int, help="补抽轮数（默认取 settings.course_topic.gleaning，0 关闭——对半省调用费）")
    args = p.parse_args()
    asyncio.run(probe(args.chunks, args.max_chars, args.model, args.gleaning))


if __name__ == "__main__":
    main()
