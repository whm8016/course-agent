"""表A 主表汇总：table_a / table_b / table_a_sweep JSON → 一张 markdown 表。

A-1 两臂可比（no_rag vs fact）：RAGChecker precision/recall/f1（含答漏率=1-recall）、
    关键要点覆盖率、可追溯率、表B 越界正确拒答率/误拒率。
A-2 fact 臂单臂绝对值：源片段 Top-1/Top-5 命中（n=有 gold 映射题数）、
    RAGAS context_precision/context_recall（取七级消融的 hybrid_rrf+rerank 臂，
    与本系统 fact 臂同一检索配置）。
A-3 fact 臂 top_k 扫描：f1 与 hallucination。

不适用项标注：闭卷臂的片段类指标（hallucination/self_knowledge/faithfulness/源片段
命中率）写「—」并注明定义原因（无检索片段：self_knowledge 恒 1、hallucination 恒
=1-precision），不编数字。

用法（backend 下）：
    python -P scripts/eval_rag/compose_table_a.py                       # 自动取 results/ 最新
    python -P scripts/eval_rag/compose_table_a.py --a <json> --b <json> --sweep <json>

产物：scripts/eval_rag/results/table_a_composed_<ts>.md（同时打印到终端）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Windows GBK 控制台兜底：正文已用 ASCII 减号，这里再防个别字符炸 print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from scripts.eval_rag import config  # noqa: E402

logger = logging.getLogger(__name__)

_ARM_LABEL = {"no_rag": "闭卷直答 (no_rag)", "fact": "本系统 (fact, PG 全流水线含精排)"}


def _latest_json(prefix: str) -> Path | None:
    """results/ 下该前缀最新的 JSON。table_a_sweep 与 table_a 前缀重叠，须互斥排除。

    按文件名倒序取「最新」（时间戳编码在文件名，与 stats.load_last_summary 同口径）——
    mtime 会被 git 恢复/云同步改动，同一批文件两种「最新」可能选中不同跑批。
    """
    exclude = "table_a_sweep_" if prefix == "table_a_" else None
    files = [
        p for p in config.RESULTS_DIR.glob(prefix + "*.json")
        if not (exclude and p.name.startswith(exclude))
    ]
    return max(files, key=lambda p: p.name) if files else None


def _latest_ragas() -> Path | None:
    """最新的可用消融 eval_summary（A-2 的 RAGAS 数字来源），按文件名降序首个可用早退。

    只认同时算出 context_precision/context_recall 且该臂没被判分异常掏空的跑批：
    RAGAS 判崩的题（NaN）被剔出均值后，全剔指标的均值是 0.0 假数字（实测
    eval_summary_20260816_041106 六个臂 27 题全 invalid → 双指标 0.0），不能进表。
    """
    for p in sorted(config.RESULTS_DIR.glob("eval_summary_*.json"),
                    key=lambda x: x.name, reverse=True):
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        arm = (data.get("avg_scores") or {}).get("hybrid_rrf+rerank") or {}
        if "context_precision" not in arm or "context_recall" not in arm:
            continue
        # 逐指标防空壳：某指标「有 invalid 且均值恰为 0.0」即视为不可信，整臂跳过
        invalid = (data.get("invalid_by_mode") or {}).get("hybrid_rrf+rerank") or {}
        if any(invalid.get(m) and arm[m] == 0.0
               for m in ("context_precision", "context_recall")):
            continue
        return p
    return None


def _load(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as e:
        logger.warning("读取 %s 失败：%s", path, e)
        return None


def _f(v, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _miss_rate(recall) -> float | None:
    """答漏率 = 1 - recall。recall 缺失（RAGChecker 未跑/失败）时返回 None 标「—」。"""
    return None if recall is None else 1 - recall


def compose(
    a: dict | None, b: dict | None, sweep: dict | None,
    ragas: dict | None, ragas_ts: str,
) -> str:
    arms = (a or {}).get("arms", {})
    no_rag, fact = arms.get("no_rag", {}), arms.get("fact", {})
    n = (a or {}).get("n", "?")
    hit = fact.get("source_hit", {})

    lines: list[str] = []
    lines.append("# 表A 主表：闭卷直答 vs 本系统")
    # 旧口径跑批警告：改动前的 table_a JSON 没有这些字段（且 citation_rate 含
    # 「拒答话术当引用」历史 bug 数字），静默复用会把废数字重新发表。
    # 须 fact 臂在本轮 modes 里才判（--skip fact 时 fact={} 不是旧口径问题）
    if "fact" in (a or {}).get("modes", []) and "source_hit" not in fact:
        lines.append("")
        lines.append("> ⚠ **警告：table_a JSON 为旧口径跑批**（fact 臂缺 source_hit 字段）。"
                     "可追溯率仍是正则修复前的数字（闭卷臂 0.2222 即历史 bug 值），"
                     "请重跑 `run_table_ab.py --table a` 后再 compose。")
    lines.append("")
    lines.append(
        f"- 生成：{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"｜course：{(a or {}).get('course', '?')}"
        f"｜评测集 n={n}（qa_dataset 人工题）"
    )
    lines.append("")
    lines.append("## A-1 两臂可比指标")
    lines.append("")
    lines.append(f"| 指标 | {_ARM_LABEL['no_rag']} | {_ARM_LABEL['fact']} |")
    lines.append("|---|---|---|")
    lines.append(f"| RAGChecker precision | {_f(no_rag.get('precision'))} | {_f(fact.get('precision'))} |")
    lines.append(f"| RAGChecker recall | {_f(no_rag.get('recall'))} | {_f(fact.get('recall'))} |")
    lines.append(f"| **RAGChecker F1** | **{_f(no_rag.get('f1'))}** | **{_f(fact.get('f1'))}** |")
    lines.append(f"| 答漏率（1-recall） | {_f(_miss_rate(no_rag.get('recall')))} | {_f(_miss_rate(fact.get('recall')))} |")
    lines.append(f"| 关键要点覆盖率 | {_f(no_rag.get('keypoint_coverage'))} | {_f(fact.get('keypoint_coverage'))} |")
    lines.append(f"| 可追溯率 | {_f(no_rag.get('citation_rate'))} | {_f(fact.get('citation_rate'))} |")

    b_arms = (b or {}).get("arms", {})
    b_no, b_fact = b_arms.get("no_rag", {}), b_arms.get("fact", {})
    if b:
        n_ood, n_in = b.get("n_ood", "?"), b.get("n_in_scope", "?")
        lines.append(f"| 越界正确拒答率（表B，n={n_ood}） | {_f(b_no.get('correct_refusal_rate'))} | {_f(b_fact.get('correct_refusal_rate'))} |")
        lines.append(f"| 误拒率（表B 课内，n={n_in}） | {_f(b_no.get('false_refusal_rate'))} | {_f(b_fact.get('false_refusal_rate'))} |")
    else:
        lines.append("| 越界正确拒答率（表B） | —（未跑） | —（未跑） |")
        lines.append("| 误拒率（表B 课内） | —（未跑） | —（未跑） |")

    lines.append("")
    lines.append("片段类指标（仅 fact 臂报；闭卷臂无检索片段，按定义不适用）：")
    lines.append("")
    lines.append("| 指标 | 闭卷直答 | 本系统 |")
    lines.append("|---|---|---|")
    lines.append(f"| hallucination | —（恒 =1-precision） | {_f(fact.get('hallucination'))} |")
    lines.append(f"| self_knowledge | —（恒为 1） | {_f(fact.get('self_knowledge'))} |")
    lines.append(f"| faithfulness | —（无检索片段不可算） | {_f(fact.get('faithfulness'))} |")

    lines.append("")
    lines.append("## A-2 本系统（fact）单臂绝对值")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 源片段 Top-1 命中 | {_f(hit.get('top1'))}（n={hit.get('n', '?')}） |")
    lines.append(f"| 源片段 Top-5 命中 | {_f(hit.get('top5'))}（n={hit.get('n', '?')}） |")
    if ragas:
        arm = ragas.get("avg_scores", {}).get("hybrid_rrf+rerank", {})
        src = f"七级消融 hybrid_rrf+rerank 臂，{ragas_ts or '?'}"
    else:
        arm, src = {}, "未找到含该臂的 eval_summary"
    lines.append(f"| RAGAS context_precision | {_f(arm.get('context_precision'), 4)}（{src}） |")
    lines.append(f"| RAGAS context_recall | {_f(arm.get('context_recall'), 4)}（{src}） |")

    lines.append("")
    lines.append(f"## A-3 top_k 敏感性（fact 臂，n={(sweep or {}).get('n', '?')}）")
    lines.append("")
    lines.append("| top_k | F1 | hallucination |")
    lines.append("|---|---|---|")
    sw = (sweep or {}).get("sweep", {})
    if sw:
        for k in sorted(sw, key=int):
            lines.append(
                f"| {k} | {_f(sw[k].get('f1'))} | {_f(sw[k].get('hallucination'))} |"
            )
    else:
        lines.append("| —（未跑 --sweep） | — | — |")

    lines.append("")
    lines.append("### 表注")
    lines.append("")
    lines.append("- 陈述级 precision/recall/f1 由 RAGChecker 判分（claim 分解+蕴含判断），"
                 "两臂同一判分模型，可比；闭卷臂的这三个数是其课程专有事实能力的直接证据。")
    lines.append("- 命中率 n 为 oracle_gold.json 有源片段映射的题数（gold 锚点覆盖 ≥0.5 才映射），"
                 "比对零 LLM。")
    lines.append("- 表B 为正则判拒答（零 LLM）：越界题拒答=对，课内题拒答=错。")
    lines.append("- 本系统 fact 臂为 PG 线全流水线（结构化切块 + dense + PG tsvector + RRF + 精排），"
                 "评测进程内 RERANK__ENABLED=true（生产默认关）。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总表A三张子表为 markdown 主表")
    parser.add_argument("--a", default=None, help="table_a JSON（默认 results/ 最新）")
    parser.add_argument("--b", default=None, help="table_b JSON（默认最新；缺则表B 行标「—」）")
    parser.add_argument("--sweep", default=None, help="table_a_sweep JSON（默认最新）")
    parser.add_argument("--ragas", default=None,
                        help="消融 eval_summary JSON（默认最新含 hybrid_rrf+rerank 臂）")
    parser.add_argument("-o", "--out", default=None, help="输出 md 路径（默认 results/ 带时间戳）")
    args = parser.parse_args()

    ragas_pick = None if args.ragas else _latest_ragas()
    a_p = Path(args.a) if args.a else _latest_json("table_a_")
    b_p = Path(args.b) if args.b else _latest_json("table_b_")
    s_p = Path(args.sweep) if args.sweep else _latest_json("table_a_sweep_")
    r_p = Path(args.ragas) if args.ragas else ragas_pick
    for p in (a_p, b_p, s_p, r_p):
        if p is not None and not p.exists():
            raise SystemExit(f"文件不存在: {p}")

    a, b, sweep, ragas = _load(a_p), _load(b_p), _load(s_p), _load(r_p)
    if a is None:
        raise SystemExit("缺少 table_a JSON（--a 指定或先跑 run_table_ab.py --table a）")

    md = compose(a, b, sweep, ragas, (ragas or {}).get("timestamp", ""))
    out = Path(args.out) if args.out else (
        config.RESULTS_DIR / f"table_a_composed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", "utf-8")
    print(f"# 输入：a={a_p} b={b_p} sweep={s_p} ragas={r_p}")
    print(f"# 产物 → {out}\n")
    print(md)


if __name__ == "__main__":
    main()
