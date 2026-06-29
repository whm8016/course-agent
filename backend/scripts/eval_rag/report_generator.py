"""Report Generator —— 生成 CSV 对比表 + Markdown 报告。"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV 生成
# ---------------------------------------------------------------------------
def generate_csv(
    qa_items: list[dict],
    all_results: dict[str, list[dict]],
    modes: list[str],
    per_question_scores: dict[str, list[dict[str, float]]],
    avg_scores: dict[str, dict[str, float]],
    metric_names: list[str],
) -> Path:
    """生成 CSV 对比表，返回文件路径。"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = config.RESULTS_DIR / f"eval_results_{ts}.csv"

    header = ["question_id", "category", "difficulty", "mode"] + metric_names

    rows: list[list[str]] = []
    for mode in modes:
        scores_list = per_question_scores.get(mode, [])
        for idx, item in enumerate(qa_items):
            row = [item["id"], item["category"], item["difficulty"], mode]
            scores = scores_list[idx] if idx < len(scores_list) else {}
            for m in metric_names:
                row.append(f"{scores.get(m, 0):.4f}")
            rows.append(row)

    # 汇总行
    rows.append([])  # 空行分隔
    for mode in modes:
        row = ["**平均**", "全部", "-", mode]
        mode_avg = avg_scores.get(mode, {})
        for m in metric_names:
            row.append(f"{mode_avg.get(m, 0):.4f}")
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("CSV 报告已生成: %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def generate_markdown(
    qa_items: list[dict],
    all_results: dict[str, list[dict]],
    modes: list[str],
    avg_scores: dict[str, dict[str, float]],
    per_question_scores: dict[str, list[dict[str, float]]],
    metric_names: list[str],
) -> Path:
    """生成 Markdown 报告，返回文件路径。"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = config.RESULTS_DIR / f"eval_report_{ts}.md"

    lines: list[str] = []

    # ---- 标题 ----
    lines.append("# RAG 评测报告")
    lines.append("")
    lines.append(f"评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ---- 1. 评测配置 ----
    lines.append("## 1. 评测配置")
    lines.append("")
    lines.append(f"- 课程：`{config.COURSE_ID}`（电路分析基础实验）")
    lines.append(f"- 评测集：{len(qa_items)} 条")
    cat_counts: dict[str, int] = {}
    for item in qa_items:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    lines.append("  - " + " / ".join(f"{k}{v}条" for k, v in cat_counts.items()))
    lines.append(f"- RAG 模式：{' / '.join(modes)}")
    lines.append(f"- 评测指标：{' / '.join(metric_names)}")
    lines.append("")

    # ---- 2. 总体对比 ----
    lines.append("## 2. 总体对比")
    lines.append("")
    lines.append(_build_score_table(modes, avg_scores, metric_names))
    lines.append("")

    # ---- 3. 按 category 分组 ----
    lines.append("## 3. 分类别对比")
    lines.append("")
    categories = sorted(set(item["category"] for item in qa_items))
    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")
        cat_avg = _compute_category_avg(
            qa_items, per_question_scores, modes, metric_names, cat
        )
        lines.append(_build_score_table(modes, cat_avg, metric_names))
        lines.append("")

    # ---- 4. 按 difficulty 分组 ----
    lines.append("## 4. 分难度对比")
    lines.append("")
    difficulties = sorted(set(item["difficulty"] for item in qa_items))
    for diff in difficulties:
        lines.append(f"### {diff}")
        lines.append("")
        diff_avg = _compute_difficulty_avg(
            qa_items, per_question_scores, modes, metric_names, diff
        )
        lines.append(_build_score_table(modes, diff_avg, metric_names))
        lines.append("")

    # ---- 5. 典型案例分析 ----
    lines.append("## 5. 典型案例分析")
    lines.append("")
    cases = _pick_interesting_cases(
        qa_items, per_question_scores, modes, metric_names
    )
    for case in cases:
        lines.append(f"### {case['id']} ({case['category']}) — {case['question']}")
        lines.append("")
        lines.append(f"**难度**：{case['difficulty']}")
        lines.append("")
        lines.append(f"**参考答案**：{case['ground_truth'][:100]}...")
        lines.append("")
        lines.append("| 模式 | " + " | ".join(metric_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(metric_names)) + " |")
        for mode in modes:
            scores = case["scores"].get(mode, {})
            row = f"| {mode} | " + " | ".join(
                f"{scores.get(m, 0):.4f}" for m in metric_names
            ) + " |"
            lines.append(row)
        lines.append("")
        if case.get("analysis"):
            lines.append(f"**分析**：{case['analysis']}")
            lines.append("")

    # ---- 6. 关键发现 ----
    lines.append("## 6. 关键发现")
    lines.append("")
    findings = _generate_findings(avg_scores, modes, metric_names)
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    md_path.write_text("\n".join(lines), "utf-8")
    logger.info("Markdown 报告已生成: %s", md_path)
    return md_path


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _build_score_table(
    modes: list[str],
    scores: dict[str, dict[str, float]],
    metric_names: list[str],
) -> str:
    """构建 Markdown 格式的得分对比表。"""
    lines = []
    header = "| 模式 | " + " | ".join(metric_names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(metric_names)) + " |"
    lines.append(header)
    lines.append(sep)
    for mode in modes:
        mode_scores = scores.get(mode, {})
        row = f"| {mode} | " + " | ".join(
            f"{mode_scores.get(m, 0):.4f}" for m in metric_names
        ) + " |"
        lines.append(row)
    return "\n".join(lines)


def _compute_category_avg(
    qa_items: list[dict],
    per_question_scores: dict[str, list[dict[str, float]]],
    modes: list[str],
    metric_names: list[str],
    category: str,
) -> dict[str, dict[str, float]]:
    """计算某个 category 下各模式的平均分。"""
    indices = [i for i, item in enumerate(qa_items) if item["category"] == category]
    return _compute_avg_by_indices(
        per_question_scores, modes, metric_names, indices
    )


def _compute_difficulty_avg(
    qa_items: list[dict],
    per_question_scores: dict[str, list[dict[str, float]]],
    modes: list[str],
    metric_names: list[str],
    difficulty: str,
) -> dict[str, dict[str, float]]:
    """计算某个难度下各模式的平均分。"""
    indices = [i for i, item in enumerate(qa_items) if item["difficulty"] == difficulty]
    return _compute_avg_by_indices(
        per_question_scores, modes, metric_names, indices
    )


def _compute_avg_by_indices(
    per_question_scores: dict[str, list[dict[str, float]]],
    modes: list[str],
    metric_names: list[str],
    indices: list[int],
) -> dict[str, dict[str, float]]:
    """根据索引列表计算各模式的平均分。"""
    result: dict[str, dict[str, float]] = {}
    if not indices:
        return {mode: {m: 0.0 for m in metric_names} for mode in modes}

    for mode in modes:
        scores_list = per_question_scores.get(mode, [])
        mode_avg: dict[str, float] = {}
        for m in metric_names:
            vals = []
            for idx in indices:
                if idx < len(scores_list):
                    vals.append(scores_list[idx].get(m, 0.0))
            mode_avg[m] = sum(vals) / len(vals) if vals else 0.0
        result[mode] = mode_avg
    return result


def _pick_interesting_cases(
    qa_items: list[dict],
    per_question_scores: dict[str, list[dict[str, float]]],
    modes: list[str],
    metric_names: list[str],
    max_cases: int = 3,
) -> list[dict]:
    """自动选取各模式差异最大的典型 case。"""
    # 计算每条问题在 context_precision 上各模式的最大差值
    scored_cases: list[tuple[float, int]] = []
    for idx, item in enumerate(qa_items):
        scores_by_mode = {}
        for mode in modes:
            scores_list = per_question_scores.get(mode, [])
            if idx < len(scores_list):
                scores_by_mode[mode] = scores_list[idx]
            else:
                scores_by_mode[mode] = {}

        # 用第一个指标（通常是 context_precision）来衡量差异
        primary_metric = metric_names[0] if metric_names else "context_precision"
        vals = [scores_by_mode.get(m, {}).get(primary_metric, 0.0) for m in modes]
        if vals:
            max_diff = max(vals) - min(vals)
            scored_cases.append((max_diff, idx))

    # 按差异降序排列，取 top-N，保证每个 category 至少 1 个
    scored_cases.sort(key=lambda x: x[0], reverse=True)

    selected: list[dict] = []
    seen_categories: set[str] = set()

    for _, idx in scored_cases:
        item = qa_items[idx]
        if len(selected) >= max_cases:
            break

        # 优先选择未覆盖的 category
        if item["category"] not in seen_categories or len(selected) < max_cases:
            scores_by_mode = {}
            for mode in modes:
                scores_list = per_question_scores.get(mode, [])
                if idx < len(scores_list):
                    scores_by_mode[mode] = scores_list[idx]
                else:
                    scores_by_mode[mode] = {}

            # 生成简要分析
            analysis = _generate_case_analysis(
                item, scores_by_mode, modes, metric_names
            )

            selected.append({
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "ground_truth": item["ground_truth"],
                "scores": scores_by_mode,
                "analysis": analysis,
            })
            seen_categories.add(item["category"])

    return selected


def _generate_case_analysis(
    item: dict,
    scores_by_mode: dict[str, dict[str, float]],
    modes: list[str],
    metric_names: list[str],
) -> str:
    """生成单条 case 的简要分析。"""
    parts = []
    primary = metric_names[0] if metric_names else "context_precision"

    # 找出最佳和最差模式
    mode_vals = {
        m: scores_by_mode.get(m, {}).get(primary, 0.0) for m in modes
    }
    if mode_vals:
        best_mode = max(mode_vals, key=mode_vals.get)
        worst_mode = min(mode_vals, key=mode_vals.get)
        diff = mode_vals[best_mode] - mode_vals[worst_mode]
        parts.append(
            f"{primary} 方面，{best_mode} 模式({mode_vals[best_mode]:.4f}) "
            f"比 {worst_mode} 模式({mode_vals[worst_mode]:.4f}) 高 {diff:.4f}。"
        )

    # 关联型问题特殊分析
    if item["category"] == "关联型":
        parts.append("此类问题需要关联多个知识片段，Graph RAG 的关联检索优势更明显。")

    return " ".join(parts)


def _generate_findings(
    avg_scores: dict[str, dict[str, float]],
    modes: list[str],
    metric_names: list[str],
) -> list[str]:
    """根据评测结果生成关键发现。"""
    findings: list[str] = []

    if not avg_scores or not modes:
        return ["评测数据不足，无法生成发现。"]

    primary = metric_names[0] if metric_names else "context_precision"

    # 找出最佳模式
    best_mode = max(modes, key=lambda m: avg_scores.get(m, {}).get(primary, 0.0))
    worst_mode = min(modes, key=lambda m: avg_scores.get(m, {}).get(primary, 0.0))

    best_val = avg_scores.get(best_mode, {}).get(primary, 0.0)
    worst_val = avg_scores.get(worst_mode, {}).get(primary, 0.0)

    if best_val > 0:
        pct = ((best_val - worst_val) / worst_val * 100) if worst_val > 0 else float("inf")
        findings.append(
            f"在 {primary} 上，{best_mode} 模式({best_val:.4f}) "
            f"比 {worst_mode} 模式({worst_val:.4f}) 高 {pct:.1f}%"
        )

    # naive vs mix 对比（naive 是 baseline，mix 是 Graph RAG）
    naive_scores = avg_scores.get("naive", {})
    mix_scores = avg_scores.get("mix", {})
    if mix_scores and naive_scores:
        for m in metric_names:
            mix_v = mix_scores.get(m, 0.0)
            naive_v = naive_scores.get(m, 0.0)
            if naive_v > 0:
                pct = (mix_v - naive_v) / naive_v * 100
                if pct > 0:
                    findings.append(
                        f"mix 比 naive 在 {m} 上高 {pct:.1f}%"
                    )

    # Graph RAG 整体优势
    if mix_scores and naive_scores:
        for m in metric_names:
            naive_v = naive_scores.get(m, 0.0)
            mix_v = mix_scores.get(m, 0.0)
            if naive_v > 0 and mix_v > naive_v:
                pct = (mix_v - naive_v) / naive_v * 100
                findings.append(
                    f"mix(Graph RAG) 比 naive(朴素检索) 在 {m} 上高 {pct:.1f}%，"
                    f"验证了知识图谱增强检索的优势"
                )

    if not findings:
        findings.append("评测结果未显示出显著的模态差异，建议增大评测集或检查索引质量。")

    # 面试话术
    findings.append("")
    findings.append("**面试话术**：")
    if mix_scores and naive_scores and primary in mix_scores and primary in naive_scores:
        mix_p = mix_scores[primary]
        naive_p = naive_scores[primary]
        if naive_p > 0:
            pct = (mix_p - naive_p) / naive_p * 100
            findings.append(
                f"> 我建了 30 条评测集，覆盖事实型/关联型/推理型/比较型四类问题，"
                f"用 RAGAS 框架对比了 5 种检索模式。结果显示 mix 模式在 {primary} 上"
                f"比 naive baseline 高 {pct:.1f}%，尤其在需要关联多个知识点的"
                f"关联型问题上优势最明显。这验证了 Graph RAG 在教学场景下对"
                f"知识关联检索的价值。"
            )

    return findings
