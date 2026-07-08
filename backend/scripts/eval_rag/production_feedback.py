"""生产问答回流：从 messages 表导出真实用户问答 + 启发式质量初筛。

============================================================
【诚实边界 - 用之前先读这段】
============================================================
这不是"用户负反馈回流"。本项目目前【没有】任何点踩 / 差评 / rating 功能，
RAG 检索命中信号（empty / retrieved_chars）也只打日志、不落 messages.metadata。
因此无法精确识别"用户判为差的回答"。

本脚本做的是【务实可落地】的两件事：
  1. 从 messages 表按 session 配对真实 (user_question, assistant_answer)，让评测集
     纳入真实问题分布——而不只用手写 30 条。
  2. 对每条 answer 做零成本启发式初筛，标出【可疑低质量候选】（answer 过短 / 含
     拒绝话术 / 含错误标记），供人工复核或喂合成器补 ground_truth。

产出是【待标注候选池】，不是可直接评测的数据集：真实问题没有标准答案（ground_truth），
而 context_precision / faithfulness 等 RAGAS 指标必须依赖 ground_truth。所以 v3 用途：
  - 人工挑值得的，补 ground_truth 后并入 v1_manual；
  - 或作为 dataset_generator 的真实问题种子。

真正的"用户负反馈闭环"需要：messages.metadata 写入 RAG 命中字段 + 前端点踩按钮，
那是独立的功能开发，不在本脚本范围。

Usage:
    # 导出某课程近 30 天的真实问答（全量 + 可疑标记）
    python -m scripts.eval_rag.production_feedback --course circuit_analysis --since-days 30

    # 只看启发式判为可疑的（人工复核优先看这批）
    python -m scripts.eval_rag.production_feedback --suspect-only --since-days 7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 让 import 找到 backend 根目录（scripts/eval_rag/ → backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
# pydantic settings 嵌套字段用双下划线（与 run_eval 一致）
os.environ.setdefault("LIGHTRAG__ENABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"

# ---------------------------------------------------------------------------
# 启发式初筛阈值与话术（零成本，不调 LLM；宁多标不漏标，交人工复核）
# ---------------------------------------------------------------------------
MIN_ANSWER_CHARS = 20

# 高置信的"没检索到 / 兜底拒绝 / 出错"话术。刻意收窄，避免误伤正常回答。
REFUSAL_PATTERNS = [
    r"我不知道",
    r"无法(?:为你|直接)?回答",
    r"没(?:有)?找到相关",
    r"未找到相关",
    r"(?:暂无|没有)相关(?:资料|信息|内容)",
    r"知识库不可用",
    r"未选择课程",
    r"检索失败",
    r"出错了",
    r"请(?:提供|补充)(?:更多|具体)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))


def classify_answer(answer: str) -> tuple[bool, str]:
    """启发式初筛：返回 (是否可疑, 原因)。空 / 过短 / 含拒绝话术 → 可疑。

    这只是粗筛，有假阳性（正常短回答被标）和假阴性（错误答案文采飞扬没被标），
    用于缩小人工复核范围，非确证质量。
    """
    a = (answer or "").strip()
    if not a:
        return True, "answer_empty"
    # 先判语义（拒绝话术），再判长度：否则"我不知道""没找到相关资料"这类短拒绝
    # 会被 too_short 截胡，丢失最高置信的"没检索到"信号，人工复核也无法按 reason 分类
    m = _REFUSAL_RE.search(a)
    if m:
        return True, f"refusal_phrase({m.group()})"
    if len(a) < MIN_ANSWER_CHARS:
        return True, f"answer_too_short({len(a)}<{MIN_ANSWER_CHARS})"
    return False, ""


def pair_user_assistant(
    msgs_by_session: dict[str, list[dict]], cutoff_ts: float = 0.0
) -> list[dict]:
    """按 session 配对 (user → 其后第一条 assistant)。

    msgs_by_session: {session_id: [{id, role, content, created_at, course_id}, ...]}
    每个 session 的列表须已按 created_at 升序（fetch 端保证）。
    """
    pairs: list[dict] = []
    for sid, smsgs in msgs_by_session.items():
        for i, m in enumerate(smsgs):
            if m["role"] != "user":
                continue
            if cutoff_ts and m["created_at"] < cutoff_ts:
                continue
            q = (m["content"] or "").strip()
            if not q:
                continue
            ans = ""
            for j in range(i + 1, len(smsgs)):
                if smsgs[j]["role"] == "assistant":
                    ans = (smsgs[j]["content"] or "").strip()
                    break
            if not ans:
                continue  # 该 user 无回答跟随（被中断 / 异常），跳过
            pairs.append({
                "question": q,
                "answer": ans,
                "course_id": m.get("course_id", ""),
                "session_id": sid,
                "created_at": m["created_at"],
                "q_msg_id": m["id"],
            })
    return pairs


def to_candidate(pair: dict) -> dict:
    """把配对转成导出条目。真实问题无标准答案 → ground_truth=None（待人工补）。"""
    suspect, reason = classify_answer(pair["answer"])
    return {
        "id": f"prod_{pair['q_msg_id']}",
        "question": pair["question"],
        "answer": pair["answer"],
        "source": "production",
        "course_id": pair["course_id"],
        "session_id": pair["session_id"],
        "created_at": datetime.fromtimestamp(
            pair["created_at"], tz=timezone.utc
        ).isoformat(),
        "suspect_low_quality": suspect,
        "suspect_reason": reason,
        "ground_truth": None,  # 待人工补；补完才能并入 v1 跑 RAGAS 指标
    }


async def fetch_qa_pairs(
    course_id: str | None, since_days: int | None, limit: int
) -> list[dict]:
    """从 messages 表拉取并配对真实 QA。碰 DB（集成路径，不在单测范围）。"""
    from sqlalchemy import select

    from core.db.database import AsyncSessionLocal, Message, Session

    cutoff_ts = time.time() - since_days * 86400 if since_days else 0.0

    async with AsyncSessionLocal() as db:
        # 先筛 session（按 course），再拉这些 session 的全部 message
        sess_q = select(Session.id, Session.course_id)
        if course_id:
            sess_q = sess_q.where(Session.course_id == course_id)
        sess_rows = (await db.execute(sess_q)).all()
        session_course = {r.id: r.course_id for r in sess_rows}
        if not session_course:
            return []

        msg_q = (
            select(Message)
            .where(Message.session_id.in_(list(session_course.keys())))
            .order_by(Message.session_id, Message.created_at.asc())
        )
        msgs = (await db.execute(msg_q)).scalars().all()

    # 聚合为 {session_id: [dict]} 喂给纯函数 pair_user_assistant
    by_session: dict[str, list[dict]] = {}
    for m in msgs:
        by_session.setdefault(m.session_id, []).append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at,
            "course_id": session_course.get(m.session_id, ""),
        })

    pairs = pair_user_assistant(by_session, cutoff_ts=cutoff_ts)
    return pairs[:limit]


async def main() -> None:
    parser = argparse.ArgumentParser(description="生产问答回流：导出真实问答 + 启发式质量初筛")
    parser.add_argument("--course", default=None, help="只导出该 course_id 的会话（不传=全部）")
    parser.add_argument("--since-days", type=int, default=None, help="只导出近 N 天（不传=不限）")
    parser.add_argument("--limit", type=int, default=1000, help="最多导出多少条 QA 对")
    parser.add_argument("--suspect-only", action="store_true", help="只导出启发式判为可疑的")
    parser.add_argument(
        "--out", default=None,
        help="输出文件（默认 datasets/v3_production_candidates.json）",
    )
    args = parser.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATASETS_DIR / "v3_production_candidates.json"

    logger.info("从 messages 表拉取真实问答 (course=%s, since_days=%s, limit=%d)...",
                args.course or "全部", args.since_days, args.limit)
    raw = await fetch_qa_pairs(args.course, args.since_days, args.limit)
    logger.info("配对得到 %d 条真实 QA 对", len(raw))

    candidates = [to_candidate(r) for r in raw]
    if args.suspect_only:
        candidates = [c for c in candidates if c["suspect_low_quality"]]
        logger.info("--suspect-only 过滤后剩 %d 条可疑候选", len(candidates))

    n_suspect = sum(1 for c in candidates if c["suspect_low_quality"])
    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), "utf-8")
    logger.info("导出完成 -> %s（共 %d 条，其中可疑 %d 条）",
                out_path, len(candidates), n_suspect)
    logger.info("【提醒】产出为待标注候选池（无 ground_truth），补 ground_truth 后并入 v1，或喂 dataset_generator。")
    logger.info("【声明】本项目无点踩/差评功能，RAG 命中信号不落库；suspect 标记是零成本启发式初筛，非用户负反馈。")


if __name__ == "__main__":
    asyncio.run(main())
