"""生成 oracle_gold.json：每道题的 ground_truth → 讲义源片段映射。

上界臂（oracle_rag）用：把题目对应源片段直接喂回答模型（DeR² 受控输入思路），
区分「检索没找到」与「找到了却没用好」。零 LLM——纯字符串覆盖比对。

映射规则（确定性，无模型参与）：
  1. 从 ground_truth 抽「锚点串」：连续 ≥4 字的中文/字母数字段（去掉标点），
     保留长度 ≥4 的段（如 "22种自动测量"、"80%"、"1-1′"）。
  2. 在摄入切块审计 JSON 的全部 chunk 里找：覆盖锚点串最多的 chunk。
     覆盖率 = 被覆盖锚点数 / 总锚点数；并列时取更短 chunk（更聚焦）。
  3. 覆盖率 < 0.5 的题不强行映射（留空，跑时警告）——gold 可能跨 chunk 或
     是多 chunk 拼接（如 a02 实验四+实验五），单片段喂入本就失真。

用法（backend 下）：
    python -P scripts/eval_rag/make_oracle_gold.py --course mycourse
    # 产物 scripts/eval_rag/oracle_gold.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from scripts.eval_rag import config  # noqa: E402

# 锚点串：连续的汉字/字母/数字/％/-/′（中文术语+量纲+编号），长度 ≥4 才有区分度
_ANCHOR_RE = re.compile(r"[一-鿿A-Za-z0-9%′'\-]{4,}")
# 纯数字/纯符号短锚点噪声过滤：全为 %′- 的不算
_NOISE_RE = re.compile(r"^[%′'\-]+$")
# 锚点过长（整句复述）在讲义原文里必然无命中——推理型 gold 是「由原文推出的结论」
# 而非原文照抄，且 gold 与原文常有同义转述（「端口的开路电压」vs「端的开路电压」）。
# 滑窗把长锚点降为 4 字短窗（步长 2），命中靠术语片段而非整句。
_MAX_ANCHOR_LEN = 4
_ANCHOR_STEP = 2


def _shorten(a: str) -> list[str]:
    """长锚点滑窗切短：窗口 4、步长 2，保留术语粒度子串。"""
    if len(a) <= _MAX_ANCHOR_LEN:
        return [a]
    out = []
    for i in range(0, len(a) - _MAX_ANCHOR_LEN + 1, _ANCHOR_STEP):
        out.append(a[i : i + _MAX_ANCHOR_LEN])
    return out


def extract_anchors(text: str) -> list[str]:
    out = []
    for m in _ANCHOR_RE.finditer(text):
        s = m.group()
        if _NOISE_RE.match(s):
            continue
        out.extend(_shorten(s))
    return out


def best_chunk_for(gold: str, chunks: list[str], min_cover: float = 0.5) -> tuple[str, float]:
    """找覆盖 gold 锚点最多的 chunk。返回 (chunk_text, coverage)；不达标返回 ("", cov)。

    单 chunk 覆盖不足时，退化为「多 chunk 拼接」：按锚点命中把 gold 切到
    不同 chunk、取覆盖锚点最多的前 3 个 chunk 按原文顺序拼接（如 a02 跨实验四+
    实验五、c01 开路短路法+两次电压法分居两个原理段，单片段本就装不下）。
    """
    anchors = extract_anchors(gold)
    if not anchors:
        return "", 0.0
    best: tuple[float, int, str] = (-1.0, 0, "")  # (coverage, -len, chunk)
    best_cov = 0.0
    for i, c in enumerate(chunks):
        cov = sum(1 for a in anchors if a in c) / len(anchors)
        if cov > best_cov or (cov == best_cov and c and (not best[2] or len(c) < len(best[2]))):
            best_cov = cov
            best = (cov, -len(c), c)
    if best_cov >= min_cover:
        return best[2], best_cov

    # 多 chunk 拼接：对每个锚点记命中它的 chunk 集合，贪心选覆盖锚点最多的
    # chunk（最多 5 个），按 chunks 原顺序拼接
    covered: set[str] = set()
    picked: list[int] = []
    idx_list = list(range(len(chunks)))
    for _ in range(5):
        scored = [
            (sum(1 for a in anchors if a in c and a not in covered), i)
            for i, c in zip(idx_list, (chunks[i] for i in idx_list))
        ]
        gain, i = max(scored, key=lambda t: (t[0], -len(chunks[t[1]])))
        if gain <= 0:
            break
        picked.append(i)
        covered.update(a for a in anchors if a in chunks[i])
    total_cov = len(covered) / len(anchors)
    if not picked or total_cov < min_cover:
        return "", total_cov
    merged = "\n\n---\n\n".join(chunks[i] for i in sorted(picked))
    return merged, total_cov


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 oracle gold 片段映射")
    parser.add_argument("--course", default="mycourse", help="摄入审计 JSON 的 course_id")
    parser.add_argument(
        "--dataset", default=None,
        help="评测集路径（默认 config.DATASET_PATH=qa_dataset.json）",
    )
    parser.add_argument("--min-cover", type=float, default=0.5)
    args = parser.parse_args()

    audit_path = _BACKEND_ROOT / "data" / "ingest_chunks" / f"course_{args.course}" / "latest.json"
    if not audit_path.exists():
        raise SystemExit(f"摄入审计 JSON 不存在: {audit_path}")
    chunks = json.loads(audit_path.read_text("utf-8"))["chunks"]
    ds_path = Path(args.dataset) if args.dataset else config.DATASET_PATH
    qa_items = json.loads(ds_path.read_text("utf-8"))

    out: dict[str, str] = {}
    report: list[str] = []
    for item in qa_items:
        gold = item.get("ground_truth", "")
        if not gold:
            report.append(f"{item['id']}: 无 ground_truth，跳过")
            continue
        chunk, cov = best_chunk_for(gold, chunks, args.min_cover)
        if chunk:
            out[item["id"]] = chunk
            report.append(f"{item['id']}: cov={cov:.2f} len={len(chunk)}")
        else:
            report.append(f"{item['id']}: cov={cov:.2f} < {args.min_cover}，不映射")

    dst = config.EVAL_DIR / "oracle_gold.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print("\n".join(report))
    print(f"映射 {len(out)}/{len(qa_items)} 题 → {dst}")


if __name__ == "__main__":
    main()
