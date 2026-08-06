"""Phase 6: eval_memory 三维 scorer 的 pytest 守卫（让评测入 CI）。

scorer 本身是 scripts/eval_memory/run.py 的判分逻辑；这里在 pytest 里复跑，确保
knowledge_update / abstention / decay 三维全过（确定性不变式，失败即回归）。
"""
import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_eval_scorers_all_pass(db):
    from core.db.database import AsyncSessionLocal

    from scripts.eval_memory import scorer

    for name, fn in scorer.SCORERS.items():
        async with AsyncSessionLocal() as s:
            passed, total, details = await fn(s)
        assert total > 0, f"{name} 无场景"
        assert passed == total, f"{name} 未全过 ({passed}/{total}): {details}"
