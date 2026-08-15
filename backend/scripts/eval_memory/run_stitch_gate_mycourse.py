"""mycourse 真实场景 L1：门控决策层（零 LLM，连真实 PostgreSQL + course_topic 图）。

与 ``run.py``（in-memory + stitch_cases 人工 seed 图）不同，本入口：
  - 不覆盖 DB__URL，走 backend/.env 真实库
  - 前置闭包从 ``course_topic_edge`` 读（mycourse 离线抽取的 34 主题 / 32 边）
  - 用例来自 ``stitch_cases_mycourse``（真实 topic_id + 真实问句）

Usage（docker exec backend）:
    python -m scripts.eval_memory.run_stitch_gate_mycourse
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def _run() -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db

    from . import config
    from .scorer import score_stitch_gate_mycourse

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            passed, total, details = await score_stitch_gate_mycourse(db)
        score = passed / total if total else 0.0
        gate = config.GATES["stitch_gate"]
        ok = score >= gate
        return {
            "suite": "mycourse",
            "stitch_gate_mycourse": {
                "score": round(score, 4),
                "passed": passed,
                "total": total,
                "details": details,
            },
            "passed": ok,
        }
    finally:
        await close_db()


def main() -> None:
    from . import config

    summary = asyncio.run(_run())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "eval_stitch_gate_mycourse.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r = summary["stitch_gate_mycourse"]
    logger.info("[eval] mycourse stitch_gate: %d/%d = %.3f → %s", r["passed"], r["total"], r["score"], out)
    if summary["passed"]:
        sys.exit(0)
    logger.warning("[FAIL] %s", r["details"])
    sys.exit(1)


if __name__ == "__main__":
    main()
