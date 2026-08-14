"""双臂对比：判定器只拼学生原话（新） vs 师生全文（旧口径）。

**历史脚本**：定稿已完成（2026-08-15，mycourse 8 段 24 对：一致率 0.7083 <
0.85 触发回退规则 → judge.py 落地「教师回复每轮截断前 100 字」折中版，
with_teacher 临时开关已删除）。本脚本因依赖已删参数无法重跑，保留作决策
记录；结果见 results/judge_context_ab.json。

真实段实测，不产假数字：取最近 N 条 memory_episode，按 session 分组，
match_topics → consolidation.split_topic_segments 切段（复用生产切分逻辑），
每段跑两次 judge_segment（with_teacher=True 旧 / False 新），同模型同温度。

只读：不写 mastery_observation、不改 episode 状态。

产出 backend/scripts/eval_memory/results/judge_context_ab.json：
- 同 (段, topic) 两臂等级一致率
- 等级分布迁移（poor/fair/good/弃权 各臂条数）
- 旧臂判 poor 的 (段, topic) 中新臂仍判 poor 的比例（poor 覆盖率）
- 引用校验失败数（dropped_bad_quote 合计）
- prompt 字符数与降幅

决策规则（先写死再跑，不事后手调）：
- 一致率 >= 0.85 且 poor 覆盖率 >= 0.9 → 采纳「只拼学生」，删 with_teacher 开关
- 不达标 → 回退「教师回答截断前 100 字」折中版（保「不对，应该是…」类纠正证据）
"""
import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from settings import get_settings  # noqa: E402,F401  (触发 .env 加载)

# 注：judge_segment 的 with_teacher 参数已在定稿时删除，judge_arm 引用它的
# 两处（with_teacher=True/False）无法再执行——见模块 docstring 的历史说明。

RESULTS_DIR = Path(__file__).parent / "results"

# 判定等级的展示顺序（弃权=unknown 单列）
_LEVEL_ORDER = ("poor", "fair", "good", "abstain")


async def load_segments(course_id: str, limit: int, max_sessions: int):
    """最近 episode 按 session 分组切段，返回 [(session_id, [(u, a), ...], seg_ep_ids)]。"""
    from sqlalchemy import select

    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from core.memory.consolidation import split_topic_segments
    from core.memory.topic_match import match_topics

    out = []
    async with AsyncSessionLocal() as db:
        # 只 select 本脚本用到的列：本地库可能缺新迁移列（topic_ids 等），只读
        # probe 不应被全列 ORM 映射绊倒
        rows = (
            await db.execute(
                select(
                    MemoryEpisode.id,
                    MemoryEpisode.session_id,
                    MemoryEpisode.user_msg,
                    MemoryEpisode.assistant_msg,
                    MemoryEpisode.created_at,
                )
                .where(
                    MemoryEpisode.course_id == course_id,
                    MemoryEpisode.user_msg != "",
                )
                .order_by(MemoryEpisode.created_at.desc())
                .limit(limit)
            )
        ).all()
        # 按时间升序回放（取的是最近 N 条，段切分要时序）
        rows = sorted(rows, key=lambda r: r.created_at)
        by_session: dict[str, list] = {}
        for ep in rows:
            by_session.setdefault(ep.session_id, []).append(ep)

        sessions = list(by_session.items())[-max_sessions:]
        n_seg = 0
        for sid, s_eps in sessions:
            hits_per_ep = []
            for ep in s_eps:
                try:
                    hits = await match_topics(
                        f"{ep.user_msg}\n{ep.assistant_msg}", course_id, db
                    )
                except Exception:
                    hits = []
                hits_per_ep.append(hits[0][0] if hits else None)
            for seg in split_topic_segments(s_eps, hits_per_ep):
                main = hits_per_ep[s_eps.index(seg[0])]
                if main is None:
                    continue  # 无主题段生产路径也不判
                out.append((sid, [(ep.user_msg, ep.assistant_msg) for ep in seg]))
                n_seg += 1
        print(f"episode={len(rows)} sessions={len(sessions)} 可判段={n_seg}")
    return out


async def judge_arm(segments, topics_by_seg, with_teacher: bool):
    """一臂全量判定。返回 [(seg_idx, JudgeResult|None, prompt_chars, segment_text)]。"""
    from core.memory.judge import judge_segment

    results = []
    for i, (_sid, turns) in enumerate(segments):
        topics = topics_by_seg[i]
        r = await judge_segment(turns, topics, with_teacher=with_teacher)
        # prompt 字符数在 judge 内部拼，这里按同口径重算（只统计对话部分）
        if with_teacher:
            dialog = "\n".join(f"学生: {u or ''}\n教师: {a or ''}" for u, a in turns)
            seg_text = "\n".join(filter(None, (f"{u}\n{a}" for u, a in turns)))
        else:
            dialog = "\n".join(f"学生: {u or ''}" for u, _a in turns)
            seg_text = "\n".join(filter(None, (u for u, _a in turns)))
        results.append((i, r, len(dialog), seg_text))
    return results


def _level_map(result) -> dict[str, str]:
    """JudgeResult → {topic_id: level}，unknown 记 'abstain'（其余 level 原样）。"""
    m = {t: "abstain" for t in (result.abstained if result else [])}
    for j in result.judgements if result else []:
        m[j.topic_id] = j.level
    return m


async def build_topics(course_id: str, segments):
    """段可判主题：复用生产口径（段内命中并集、同主题取最大 sim、降序前 3）。"""
    from core.db.database import AsyncSessionLocal, CourseTopic
    from core.memory.topic_match import match_topics
    from sqlalchemy import select

    topics_by_seg = []
    async with AsyncSessionLocal() as db:
        meta_rows = (
            await db.execute(
                select(CourseTopic.topic_id, CourseTopic.label,
                       CourseTopic.definition)
                .where(CourseTopic.course_id == course_id)
            )
        ).all()
        meta = {r[0]: (r[1] or "", r[2] or "") for r in meta_rows}
        for sid, turns in segments:
            best_sim: dict[str, float] = {}
            for u, a in turns:
                try:
                    hits = await match_topics(f"{u}\n{a}", course_id, db)
                except Exception:
                    hits = []
                for t, sim in hits:
                    if t not in best_sim or sim > best_sim[t]:
                        best_sim[t] = sim
            seg_topics = sorted(best_sim, key=lambda t: -best_sim[t])[:3]
            topics_by_seg.append(
                [(t, *meta[t]) for t in seg_topics if t in meta]
            )
    return topics_by_seg


async def run(args) -> dict:
    """单事件循环内跑完全部异步步骤：AsyncSessionLocal 模块级连接池绑首次
    事件循环，跨 asyncio.run() 复用会在 Windows proactor 上崩。"""
    print(f"LLM text_model: {get_settings().llm.text_model}")
    segments = await load_segments(args.course, args.limit, args.sessions)
    if not segments:
        print("无可判段（episode 不足或全部无主题命中）")
        return {}

    topics_by_seg = await build_topics(args.course, segments)
    # 无可判主题的段跳过（与生产一致：judge 返回 None 不算判定失败）
    keep = [i for i, t in enumerate(topics_by_seg) if t]
    segments = [segments[i] for i in keep]
    topics_by_seg = [topics_by_seg[i] for i in keep]
    print(f"进入对比的段数: {len(segments)}")

    t0 = time.time()
    old_arm = await judge_arm(segments, topics_by_seg, with_teacher=True)
    new_arm = await judge_arm(segments, topics_by_seg, with_teacher=False)
    print(f"双臂判定耗时 {time.time() - t0:.0f}s")

    # 逐 (段, topic) 对齐
    pairs = []  # (seg_idx, topic_id, old_level, new_level)
    for (i, old_r, _, _), (_, new_r, _, _) in zip(old_arm, new_arm):
        om, nm = _level_map(old_r), _level_map(new_r)
        for tid in om.keys() | nm.keys():
            pairs.append((i, tid, om.get(tid, "miss"), nm.get(tid, "miss")))

    return _build_report(args, segments, pairs, old_arm, new_arm)


def _build_report(args, segments, pairs, old_arm, new_arm) -> dict:

    agree = sum(1 for _, _, o, n in pairs if o == n)
    agreement = agree / len(pairs) if pairs else 0.0
    old_poor = [(i, t) for i, t, o, _ in pairs if o == "poor"]
    poor_kept = sum(
        1 for i, t, o, n in pairs if o == "poor" and n == "poor"
    )
    poor_coverage = poor_kept / len(old_poor) if old_poor else None

    dist_old = Counter(o for _, _, o, _ in pairs)
    dist_new = Counter(n for _, _, n, _ in pairs)
    chars_old = sum(c for _, _, c, _ in old_arm)
    chars_new = sum(c for _, _, c, _ in new_arm)

    report = {
        "course": args.course,
        "model": get_settings().llm.text_model,
        "n_segments": len(segments),
        "n_pairs": len(pairs),
        "agreement": round(agreement, 4),
        "poor_coverage": None if poor_coverage is None else round(poor_coverage, 4),
        "old_poor_pairs": len(old_poor),
        "level_dist_old": {k: dist_old.get(k, 0) for k in (*_LEVEL_ORDER, "miss")},
        "level_dist_new": {k: dist_new.get(k, 0) for k in (*_LEVEL_ORDER, "miss")},
        "bad_quote_old": sum(r.dropped_bad_quote if r else 0 for _, r, _, _ in old_arm),
        "bad_quote_new": sum(r.dropped_bad_quote if r else 0 for _, r, _, _ in new_arm),
        "judge_none_old": sum(1 for _, r, _, _ in old_arm if r is None),
        "judge_none_new": sum(1 for _, r, _, _ in new_arm if r is None),
        "dialog_chars_old": chars_old,
        "dialog_chars_new": chars_new,
        "char_reduction": round(1 - chars_new / chars_old, 4) if chars_old else None,
        "disagree_pairs": [
            {"seg": i, "topic": t, "old": o, "new": n}
            for i, t, o, n in pairs if o != n
        ][:50],
    }

    # 写死决策规则
    ok_agree = agreement >= 0.85
    ok_poor = poor_coverage is not None and poor_coverage >= 0.9
    decision = (
        "ADOPT_STUDENT_ONLY"
        if ok_agree and ok_poor
        else "FALLBACK_TRUNCATED_TEACHER"
    )
    report["decision"] = decision
    report["decision_rule"] = (
        "agreement >= 0.85 且 poor_coverage >= 0.9 → 采纳只拼学生；否则回退教师截断版"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", default="mycourse")
    parser.add_argument("--limit", type=int, default=200, help="最近 N 条 episode")
    parser.add_argument("--sessions", type=int, default=5, help="最多取最后几个 session")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    if not report:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "judge_context_ab.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("disagree_pairs",)}, ensure_ascii=False, indent=2))
    print(f"\n>> 决策: {report['decision']}  (明细已写 {out_path})")


if __name__ == "__main__":
    main()
