r"""诊断探针：验证「从讲义切块抽课程主题」这条 LLM 调用链是否稳定、抽出的主题是否像样。

与 ``core.memory.course_topic_builder.build_topic_graph`` 的区别——本脚本**只测不灌**：
  - 不写 course_topic 表、不导出 JSON、不抽前置边、不算 embedding
  - 逐块打印每次 LLM 调用的全貌（字符数 / prompt_token / finish_reason /
    content 长度 / reasoning_token / 抽到几个主题），失败不中断
  - 复用 core.memory.course_topic_builder 的 ``_pack_blocks`` + ``_EXTRACT_TOPICS_PROMPT``，
    测的就是真代码路径，不是另写一套

跑通它回答三个问题：
  1. LLM 调用稳定率（N 块成功几块）
  2. 挂的是哪些块、什么原因（content 空？输出被 reasoning 截？熔断器？）
  3. 抽出的主题质量（看末尾 label 清单，人工核是否像「可考核知识点」）

用法（在配了真实 LLM 的环境，如 docker exec backend）：
    python -m scripts.eval_memory.probe_extract_topics --chunks <adapted.json>
    python -m scripts.eval_memory.probe_extract_topics --chunks <adapted.json> --max-chars 3000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

# 仿 run.py：连真实库前先兜底环境变量（脚本可独立跑）
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

# 复用真代码的分块逻辑 + prompt，保证测的就是生产路径（抽取逻辑已下沉 core）
from core.memory.course_topic_builder import _EXTRACT_TOPICS_PROMPT, _pack_blocks  # noqa: E402

# WARNING 级：压掉 INFO 噪音，但保留 CircuitBreaker 的 WARNING（看熔断器状态）
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def probe(chunks_path: str, max_chars: int, model: str | None = None) -> None:
    from core.db.database import init_db
    from core.llm.llm import client as llm
    from settings import get_settings

    await init_db()
    s = get_settings()
    model = model or s.course_topic.extract_model or s.llm.text_model

    raw = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw

    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    total_blocks = sum(len(_pack_blocks(g, max_chars)) for g in by_section.values())
    print(f"\nmodel={model}  章节={len(by_section)}  切块后总块数={total_blocks}  max_chars={max_chars}")
    print("=" * 92)
    print(f"{'#':<4}{'章节':<22}{'字符':>6}{'pt':>6}{'ct':>6}{'(reason)':>9}"
          f"{'finish':<9}{'主题数':>6}  备注")
    print("-" * 92)

    all_labels: list[str] = []
    n_ok = n_fail = n_trunc = n_topics = 0
    i = 0
    for sec, group in by_section.items():
        for block in _pack_blocks(group, max_chars):
            i += 1
            text = "\n".join(ck["text"] for ck in block)
            prompt = f"章节：{sec}\n\n{text}\n\n{_EXTRACT_TOPICS_PROMPT}"
            try:
                resp = await llm.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=4096, stream=False,
                )
                ch = resp.choices[0]
                msg = ch.message
                raw_c = (msg.content or "").strip()
                raw_c = re.sub(r"^```json\s*", "", raw_c)
                raw_c = re.sub(r"\s*```$", "", raw_c)
                data = json.loads(raw_c) if raw_c else None
                topics = (data or {}).get("topics", []) if isinstance(data, dict) else []
                labels = [str(t.get("label") or "").strip() for t in topics if str(t.get("label") or "").strip()]
                all_labels.extend(labels)

                u = resp.usage
                rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                note = ""
                if ch.finish_reason == "length":
                    n_trunc += 1
                    note = "⚠输出被截(reasoning吃满4096)"
                elif not raw_c:
                    note = "⚠content空(json解析跳过)"
                n_ok += 1
                n_topics += len(labels)
                print(f"{i:<4}{sec[:20]:<22}{len(text):>6}{u.prompt_tokens:>6}"
                      f"{u.completion_tokens:>6}{rt:>9}{ch.finish_reason:<9}{len(labels):>6}  {note}")
            except Exception as exc:  # 单块失败不中断，继续下一块
                n_fail += 1
                print(f"{i:<4}{sec[:20]:<22}{'':>6}{'':>6}{'':>6}{'':>9}{'FAIL':<9}{'':>6}  {exc}")

    print("=" * 92)
    print(f"汇总：{n_ok} 块成功 / {n_fail} 块失败 / {n_trunc} 块输出被截，共抽到 {n_topics} 个主题（去重前）")
    if n_trunc:
        print(f"  ⚠ 有 {n_trunc} 块 finish=length：reasoning 吃满 4096 输出预算 → 调小 --max-chars 重测")
    print("\n抽到的全部主题 label（人工核：是「可考核知识点」还是「小标题/实验器材」？）：")
    for j, lb in enumerate(all_labels, 1):
        print(f"  {j:>3}. {lb}")


def main() -> None:
    p = argparse.ArgumentParser(description="诊断探针：测「切块抽课程主题」LLM 调用链（不灌库）")
    p.add_argument("--chunks", required=True, help="适配后的切块审计 JSON 路径")
    p.add_argument("--max-chars", type=int, default=5000, help="单次调用喂给 LLM 的章节字符上限")
    p.add_argument("--model", help="覆盖 text_model（建议 deepseek-v4-flash）")
    args = p.parse_args()
    asyncio.run(probe(args.chunks, args.max_chars, args.model))


if __name__ == "__main__":
    main()
