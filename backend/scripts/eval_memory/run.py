"""L3 记忆评测入口。

Usage:
    python -m scripts.eval_memory.run

跑三维 scorer（knowledge_update / abstention / decay）→ 过 config.GATES 门禁 → 落盘 summary。
门禁不达标 exit 1（exit 1 = 门禁不达标，非评测崩溃；与 eval_capabilities 一致）。

自包含：强制 in-memory SQLite，不触碰真实数据，无需 LLM，可入 CI。
对照 LongMemEval（ICLR 2025）的 knowledge updates / abstention 题型，LOCOMO 仅用于
纵向自比（各家自报口径打架，不跨系统横评）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

# 强制 in-memory SQLite + TESTING，必须在 import 任何 backend 模块前设置（同 tests/conftest.py）
os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def _run_all() -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db

    from . import config
    from .scorer import SCORERS

    await init_db()
    results: dict[str, dict] = {}
    try:
        for dim, fn in SCORERS.items():
            async with AsyncSessionLocal() as db:
                passed, total, details = await fn(db)
            score = passed / total if total else 0.0
            results[dim] = {"score": round(score, 4), "passed": passed, "total": total, "details": details}
            logger.info("[eval_memory] %s: %d/%d = %.3f", dim, passed, total, score)
    finally:
        await close_db()

    failures: list[str] = []
    for dim, gate in config.GATES.items():
        score = results[dim]["score"]
        if score < gate:
            failures.append(f"{dim}: {score:.3f} < 门禁 {gate}")
    return {"results": results, "passed": not failures, "failures": failures}


def main() -> None:
    from . import config

    summary = asyncio.run(_run_all())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "eval_memory.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary["passed"]:
        logger.info("[PASS] eval_memory 三维门禁全达标 → %s", out)
        sys.exit(0)
    else:
        logger.warning("[FAIL] eval_memory 门禁不达标：%s → %s", summary["failures"], out)
        sys.exit(1)


if __name__ == "__main__":
    main()
