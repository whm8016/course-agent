"""L2 摘要 v2 评测单 case 执行器：新 v2 管线 vs 旧 v1 基线，断言对比。

新管线：_merge_items（slot key 裁决）+ _evict_by_budget（salience 淘汰）+ _render_items。
旧基线：忠实复刻改造前的 _merge_structured（精确去重 + combined[-5:] 硬截断）+
_format_structured，且无 resolved 机制（open_question 不会被消除）。

每个 case 的 increments.items 假定 LLM 已正确抽取，eval 只验合并/淘汰/渲染（本次改造部分）。
ts 由 config.TS_BASE + i*TS_STEP 逐增量递增（代码盖章，不让 LLM 出时间戳，对齐实现）。
"""
from __future__ import annotations

from typing import Any

from . import config

# ---------------------------------------------------------------------------
# 新 v2 管线（直接复用生产纯函数）
# ---------------------------------------------------------------------------

def _run_new(case: dict) -> dict:
    """跑新 v2 管线：逐增量 merge -> evict -> render。返回渲染文本 + 统计。"""
    from core.memory.session_summary import (
        SummaryItem, _merge_items, _evict_by_budget, _render_items,
        _normalize_key, _derive_key, _estimate_tokens,
    )

    increments = case.get("increments", [])
    items: list = []
    for i, inc in enumerate(increments):
        ts = config.TS_BASE + i * config.TS_STEP
        new_items = []
        for ri in inc.get("items", []):
            k = str(ri.get("k", "")).strip().lower()
            t = str(ri.get("t", "")).strip()
            if not t:
                continue
            key = _normalize_key(str(ri.get("key", ""))) or _derive_key(t)
            new_items.append(SummaryItem(k=k, key=key, t=t, ts=ts, n=1))
        items = _merge_items(items, new_items, inc.get("resolved", []))

    now_ts = config.TS_BASE + max(0, len(increments) - 1) * config.TS_STEP
    kept = _evict_by_budget(
        items,
        token_budget=config.TOKEN_BUDGET,
        max_per_kind=config.MAX_PER_KIND,
        now_ts=now_ts,
        half_life_s=config.HALF_LIFE_S,
    )
    rendered = _render_items(kept, now_ts=now_ts, half_life_s=config.HALF_LIFE_S)
    token_total = sum(_estimate_tokens(it.t) for it in kept)
    return {
        "rendered": rendered,
        "items_count": len(kept),
        "token_total": token_total,
    }


# ---------------------------------------------------------------------------
# 旧 v1 基线（忠实复刻改造前逻辑，用于对比）
# ---------------------------------------------------------------------------

_V1_KEYS = ("topics", "decisions", "facts", "open_questions", "action_items")
_V2_TO_V1 = {
    "topic": "topics",
    "decision": "decisions",
    "fact": "facts",
    "open_question": "open_questions",
    "next_step": "action_items",
    # misconception / intent 是 v2 新增类，v1 无对应 -> 基线无法表达
}
_BASELINE_TITLES = (
    ("会话主题", "topics"),
    ("关键结论与决定", "decisions"),
    ("确认的事实", "facts"),
    ("未解决的问题", "open_questions"),
    ("约定与后续", "action_items"),
)


def _baseline_normalize(s: str) -> str:
    return (s or "").strip().lower().rstrip("。.；;，,、:：")


def _baseline_merge(existing: dict, new: dict) -> dict:
    """复刻旧 _merge_structured：归一化精确去重 + combined[-5:] 硬截断（丢最旧）。"""
    merged: dict[str, list[str]] = {}
    for key in _V1_KEYS:
        old = existing.get(key) if isinstance(existing.get(key), list) else []
        newv = new.get(key) if isinstance(new.get(key), list) else []
        seen: set[str] = set()
        combined: list[str] = []
        for item in [*old, *newv]:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item:
                continue
            norm = _baseline_normalize(item)
            if norm in seen:
                continue
            seen.add(norm)
            combined.append(item)
        merged[key] = combined[-5:]  # 旧硬编码：每类最多 5，超限丢最旧
    return merged


def _baseline_render(d: dict) -> str:
    """复刻旧 _format_structured。"""
    parts: list[str] = []
    for title, key in _BASELINE_TITLES:
        items = d.get(key) if isinstance(d.get(key), list) else []
        if not items:
            continue
        body = "\n".join(f"- {it}" for it in items)
        parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts).rstrip()


def _run_baseline(case: dict) -> dict:
    """跑旧 v1 基线。applicable=False 表示 case 用了 v2 新类（基线无法表达，对比标 N/A）。"""
    merged: dict[str, list[str]] = {k: [] for k in _V1_KEYS}
    applicable = True
    for inc in case.get("increments", []):
        inc_dict: dict[str, list[str]] = {k: [] for k in _V1_KEYS}
        for it in inc.get("items", []):
            v1k = _V2_TO_V1.get(str(it.get("k", "")).strip().lower())
            if v1k is None:
                applicable = False  # misconception/intent 等新类 -> 基线无法表达
                continue
            inc_dict[v1k].append(str(it.get("t", "")))
        merged = _baseline_merge(merged, inc_dict)
        # 旧实现无 resolved -> open_question 不会被消除（这是改造点之一）
    return {"rendered": _baseline_render(merged), "applicable": applicable}


# ---------------------------------------------------------------------------
# 断言检查
# ---------------------------------------------------------------------------

def _check_assertions(case: dict, rendered: str, token_total: int) -> tuple[bool, list[str]]:
    """返回 (全通过, 失败原因列表)。"""
    failures: list[str] = []
    for sub in case.get("must_contain", []):
        if sub not in rendered:
            failures.append(f"must_contain 缺失: {sub!r}")
    for sub in case.get("must_not_contain", []):
        if sub in rendered:
            failures.append(f"must_not_contain 仍存在: {sub!r}")
    if case.get("expect_empty") and rendered != "":
        failures.append(f"expect_empty 但渲染非空 (len={len(rendered)})")
    if case.get("check_budget") and token_total > config.TOKEN_BUDGET:
        failures.append(f"check_budget 超预算: {token_total} > {config.TOKEN_BUDGET}")
    return (not failures, failures)


def run_case(case: dict) -> dict[str, Any]:
    """单 case：跑新管线 + 基线，分别断言，组装记录。"""
    new = _run_new(case)
    new_pass, new_failures = _check_assertions(case, new["rendered"], new["token_total"])

    base = _run_baseline(case)
    if base["applicable"]:
        base_pass, base_failures = _check_assertions(case, base["rendered"], new["token_total"])
        base_status = "pass" if base_pass else "fail"
    else:
        base_pass, base_failures = None, ["N/A: case 用了 v2 新类（misconception/intent），基线无法表达"]
        base_status = "n/a"

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "desc": case.get("desc"),
        "new_status": "pass" if new_pass else "fail",
        "new_failures": new_failures,
        "baseline_status": base_status,
        "baseline_failures": base_failures,
        "items_count": new["items_count"],
        "token_total": new["token_total"],
        "new_rendered": new["rendered"],
        "baseline_rendered": base["rendered"],
    }


__all__ = ["run_case"]
