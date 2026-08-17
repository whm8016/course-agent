"""阶段5评测脚本的 CI 冒烟：抽取对比框架可跑、指标合理。

旧 test_stitch_e2e_three_arms（裸调旧 run()）已删：run_stitch_e2e 重写为表E
e2e（stitch_for_turn 全链，需 embedding 凭据），接线自检移至
tests/test_stitch_e2e_wiring.py（打桩 embed 单例，离线跑同一条 run()）。
"""
import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_extract_compare_closed_set_arm(db):
    from scripts.eval_memory import run_extract_compare

    out = await run_extract_compare._run(["closed_set"])
    arm = out["closed_set"]
    if arm.get("topic_accuracy") is None:
        pytest.skip("embedder 不可用（无网络/key），离线跳过")
    assert 0.0 <= arm["topic_accuracy"] <= 1.0
    assert arm["hit"] + len(arm["details"]) <= arm["total"]
