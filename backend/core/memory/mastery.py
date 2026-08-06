"""L3 mastery 层：知识点掌握度（course 维度），追加观测 + 读时指数衰减。

append_mastery：从抽取的知识点追加观测（不覆盖历史——blend mastery/risk + observation_count
累加 + evidence_episode_ids 追加）。course 隔离修 users.knowledge_graph 的跨课程污染。

get_mastery_context：读时按 last_observed_at 做指数衰减（effective_risk = risk · exp(-λ·age)），
取 risk 最高的 top-N 薄弱点拼成 ≤300 字注入 prompt（Phase 4b 用）。软衰减只影响排序，
不物理删除——三个月前的错误概念正是诊断反复性错误的关键证据（Graphiti bi-temporal）。
"""
from __future__ import annotations

import hashlib
import logging
import math
import time

logger = logging.getLogger(__name__)

# 读时衰减系数（按天）：0.01 → 半衰期约 69 天。旧错误概念软降权，不物理删除。
_DECAY_LAMBDA_PER_DAY = 0.01


def _clamp(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _delta(v) -> float:
    """delta 原值（可负），仅 float 转换不钳制——负 delta 表示退步/改善，钳掉会丢信号。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _kp_id_from(kp: dict) -> str:
    entity_id = str(kp.get("entity_id") or "").strip()
    if entity_id:
        return entity_id
    label = str(kp.get("label") or "").strip()
    return "kp:" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]


async def append_mastery(
    db, user_id: str, course_id: str, knowledge_points: list, episode_id: str
) -> int:
    """从抽取的知识点追加掌握度观测（不覆盖历史）。返回写入的观测数。

    支持两种抽取格式：
    - catalog(delta)：{entity_id, label, mastery_delta, risk_delta} → 累加 delta
    - fallback(绝对值)：{label, mastery, risk} → 与现有值平均
    幂等键 (user_id, course_id, kp_id)：已有行则 blend+count++，否则新建。
    """
    from core.db.database import KnowledgeMastery
    from sqlalchemy import select

    if not knowledge_points:
        return 0
    now = time.time()
    n = 0
    for kp in knowledge_points:
        label = str(kp.get("label") or "").strip()
        if not label:
            continue
        kp_id = _kp_id_from(kp)
        is_delta = "mastery_delta" in kp or "risk_delta" in kp
        m_delta = _delta(kp.get("mastery_delta")) if is_delta else 0.0
        r_delta = _delta(kp.get("risk_delta")) if is_delta else 0.0
        m_abs = _clamp(kp.get("mastery"), 0.5) if not is_delta else 0.5
        r_abs = _clamp(kp.get("risk"), 0.5) if not is_delta else 0.5

        row = (
            (
                await db.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == user_id,
                        KnowledgeMastery.course_id == course_id,
                        KnowledgeMastery.kp_id == kp_id,
                    )
                )
            )
            .scalars()
            .first()
        )

        if row is not None:
            # 追加观测：delta 累加 / 绝对值平均；evidence 追加（不覆盖历史）
            if is_delta:
                row.mastery = _clamp(row.mastery + m_delta, 0.5)
                row.risk = _clamp(row.risk + r_delta, 0.5)
            else:
                row.mastery = round((row.mastery + m_abs) / 2, 3)
                row.risk = round((row.risk + r_abs) / 2, 3)
            row.observation_count = (row.observation_count or 1) + 1
            row.last_observed_at = now
            ev = list(row.evidence_episode_ids or [])
            if episode_id and episode_id not in ev:
                ev.append(episode_id)
            row.evidence_episode_ids = ev[-20:]
        else:
            mastery_v = _clamp(0.5 + m_delta, 0.5) if is_delta else m_abs
            risk_v = _clamp(0.5 + r_delta, 0.5) if is_delta else r_abs
            db.add(
                KnowledgeMastery(
                    user_id=user_id,
                    course_id=course_id,
                    kp_id=kp_id,
                    label=label,
                    mastery=mastery_v,
                    risk=risk_v,
                    observation_count=1,
                    first_observed_at=now,
                    last_observed_at=now,
                    evidence_episode_ids=[episode_id] if episode_id else [],
                )
            )
        n += 1
    await db.commit()
    return n


async def get_mastery_context(
    db, user_id: str, course_id: str, *, top_n: int = 5, max_chars: int = 300
) -> str:
    """读时衰减后取 risk 最高的 top-N 薄弱点，拼成注入文本（Phase 4b 用）。无数据则空串。"""
    from core.db.database import KnowledgeMastery
    from sqlalchemy import select

    rows = (
        (
            await db.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == user_id,
                    KnowledgeMastery.course_id == course_id,
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return ""

    now = time.time()
    scored = []
    for r in rows:
        age_days = max(0.0, (now - (r.last_observed_at or now)) / 86400.0)
        decay = math.exp(-_DECAY_LAMBDA_PER_DAY * age_days)
        scored.append((r, r.risk * decay))

    # 薄弱点：衰减后 effective_risk 高优先；risk<0.5 的不算薄弱
    scored.sort(key=lambda x: -x[1])
    weak = [(r, eff) for (r, eff) in scored if r.risk >= 0.5][:top_n]
    if not weak:
        return ""

    lines = ["## 该生掌握度（薄弱点优先，注意诊断反复性错误）"]
    for r, _eff in weak:
        lines.append(f"- {r.label}：偏薄弱(风险{r.risk:.2f}，{r.observation_count}次观测)")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


__all__ = ["append_mastery", "get_mastery_context"]
