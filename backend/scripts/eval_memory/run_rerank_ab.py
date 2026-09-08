"""双臂对比：mem0 读侧记忆注入，精排关（向量序截断） vs 精排开（qwen3-rerank 重排）。

真实数据实测（用户执行）：取最近 N 条 memory_episode 的 user_msg 作 query，按生产
口径 match_topics 算 matched_topic，每条 query 跑两遍 build_memory_context——
  A 臂 rerank_enabled=False（现状） / B 臂 rerank_enabled=True（P0 新路径），
两臂同 query 同候选池，只差精排。文本不同的对交给 LLM pairwise judge（A/B/TIE）。

只读：不写任何表。产出 backend/scripts/eval_memory/results/rerank_ab.json：
- identical（两臂文本相同=精排未改变注入，候选未超上限或重排后头部不变）占比
- 差异对中 A/B/TIE 分布与样例
- 各臂注入条数/字符数均值（精排是否换来更聚焦的注入）

已知偏差：judge 输入 A 固定为关精排臂、B 固定为开精排臂（不换序），位置偏好看
报告时留意 TIE 占比。决策规则（先写死再跑）：B 胜率（B/(A+B+TIE 差异对)）>= 0.5
且 A 胜率 <= 0.2 → 建议开默认；否则维持默认关。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from settings import get_settings  # noqa: E402,F401  (触发 .env 加载)

RESULTS_DIR = Path(__file__).parent / "results"

_JUDGE_PROMPT = """你是教学智能体「记忆注入质量」的评测员。给定学生本轮提问与两个版本的注入记忆文本（A/B），判断哪个版本对回答该问题更有用——包含更多与问题相关的记忆、更少无关噪声。两版同样有用或同样无用输出 TIE。只输出 A、B 或 TIE，不要输出其他内容。"""


async def load_queries(course_id: str, limit: int, max_users: int):
    """最近 episode 的 (user_id, user_msg, matched_topic) 列表，去重 user 保前 max_users 个。"""
    from sqlalchemy import select

    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from core.memory.topic_match import match_topics

    out = []
    seen_users: set[str] = set()
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(
                    MemoryEpisode.user_id,
                    MemoryEpisode.user_msg,
                )
                .where(
                    MemoryEpisode.course_id == course_id,
                    MemoryEpisode.user_msg != "",
                )
                .order_by(MemoryEpisode.created_at.desc())
                .limit(limit)
            )
        ).all()
        rows = list(reversed(rows))  # 时间升序回放（就近的留最后，不截断用户）
        for user_id, user_msg in rows:
            if len(seen_users) >= max_users and user_id not in seen_users:
                continue
            try:
                hits = await match_topics(user_msg, course_id, db)
            except Exception:
                hits = []
            out.append((user_id, user_msg, hits[0][0] if hits else ""))
            seen_users.add(user_id)
        print(f"episode={len(rows)} users={len(seen_users)} 查询数={len(out)}")
    return out


async def build_arm(user_id: str, query: str, course_id: str, matched_topic: str, rerank: bool):
    """单臂注入：直接拨 settings.mem0.rerank_enabled（脚本独占进程，改完全局生效）。"""
    from core.memory.mem0_client import build_memory_context

    get_settings().mem0.rerank_enabled = rerank
    return await build_memory_context(
        user_id, query, course_id=course_id, matched_topic=matched_topic
    )


async def judge(query: str, text_a: str, text_b: str) -> str:
    """LLM pairwise judge：返回 A / B / TIE（非法输出归 TIE）。"""
    import re

    from core.llm.llm import chat_complete

    user_message = (
        f"学生提问：{query}\n\n=== A 版注入 ===\n{text_a}\n\n=== B 版注入 ===\n{text_b}"
    )
    out = await chat_complete(
        _JUDGE_PROMPT, [], user_message, temperature=0.0, max_tokens=16,
        langsmith_name="mem0_rerank_ab_judge",
    )
    m = re.search(r"\b(A|B|TIE)\b", (out or "").upper())
    return m.group(1) if m else "TIE"


def _strip_header(ctx_text: str) -> str:
    """剥掉注入 header（两臂逐字相同，留着只会稀释 judge 注意力）。

    header 固定 4 行（标题 + 三条禁令），与 build_memory_context 的拼接顺序耦合，
    该函数变了这里要跟着改。
    """
    return "\n".join(ctx_text.splitlines()[4:])


async def run(args) -> dict:
    print(f"LLM text_model: {get_settings().llm.text_model}")
    # fail-fast：开关开了但 rerank 基建不可用（缺 EMBEDDING__API_KEY 或 RERANK__ENABLED
    # =false）时 B 臂与 A 臂全同，跑完只能得到 identical=100%，提前报错省时间。
    from core.rag.rerank import build_rerank_fn

    if build_rerank_fn() is None:
        print("[!] build_rerank_fn() 返回 None（缺 EMBEDDING__API_KEY 或 RERANK__ENABLED=false）：B 臂无精排可评，退出。")
        return {}

    queries = await load_queries(args.course, args.limit, args.users)
    if not queries:
        print("无可用查询（episode 不足）")
        return {}

    n_identical = 0
    verdicts = {"A": 0, "B": 0, "TIE": 0}
    samples: list[dict] = []
    chars = {"A": 0, "B": 0}
    t0 = time.time()
    for user_id, query, matched_topic in queries:
        try:
            ctx_a = await build_arm(user_id, query, args.course, matched_topic, rerank=False)
            ctx_b = await build_arm(user_id, query, args.course, matched_topic, rerank=True)
        except Exception as exc:
            print(f"[!] query 跳过（检索失败）: {exc}")
            continue
        if not ctx_a and not ctx_b:
            continue  # 两臂都无记忆（该用户无历史），不进统计
        body_a, body_b = _strip_header(ctx_a), _strip_header(ctx_b)
        chars["A"] += len(body_a)
        chars["B"] += len(body_b)
        if body_a == body_b:
            n_identical += 1
            continue
        v = await judge(query, body_a, body_b)
        verdicts[v] += 1
        if len(samples) < args.samples:
            samples.append(
                {"user": user_id, "query": query[:100], "verdict": v,
                 "A": body_a[:300], "B": body_b[:300]}
            )
    print(f"双臂 + judge 耗时 {time.time() - t0:.0f}s")

    n_judged = sum(verdicts.values())
    b_win = verdicts["B"] / n_judged if n_judged else 0.0
    a_win = verdicts["A"] / n_judged if n_judged else 0.0
    decision = (
        "SUGGEST_ENABLE_DEFAULT"
        if b_win >= 0.5 and a_win <= 0.2
        else "KEEP_DISABLED"
    )
    return {
        "course": args.course,
        "model": get_settings().llm.text_model,
        "n_queries": len(queries),
        "n_identical": n_identical,
        "n_judged": n_judged,
        "verdicts": verdicts,
        "b_win_rate": round(b_win, 4),
        "a_win_rate": round(a_win, 4),
        "avg_chars_A": round(chars["A"] / max(len(queries), 1), 1),
        "avg_chars_B": round(chars["B"] / max(len(queries), 1), 1),
        "decision": decision,
        "decision_rule": "差异对中 B(精排) 胜率 >= 0.5 且 A 胜率 <= 0.2 → 建议开默认；否则维持默认关",
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", default="mycourse")
    parser.add_argument("--limit", type=int, default=100, help="最近 N 条 episode")
    parser.add_argument("--users", type=int, default=5, help="最多覆盖几个 user")
    parser.add_argument("--samples", type=int, default=20, help="报告里保留的样例对数")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    if not report:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "rerank_ab.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    print(f"\n>> 决策: {report['decision']}  (样例已写 {out_path})")


if __name__ == "__main__":
    main()
