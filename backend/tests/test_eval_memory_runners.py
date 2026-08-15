"""阶段5评测脚本的 CI 冒烟：三臂 e2e + 抽取对比框架可跑、指标合理。"""
import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_stitch_e2e_three_arms():
    from scripts.eval_memory import run_stitch_e2e

    summary = await run_stitch_e2e.run()
    for arm in ("baseline_none", "baseline_unconditional", "gate"):
        assert arm in summary
        assert 0.0 <= summary[arm]["balanced_accuracy"] <= 1.0
    # 门控臂应显著优于基线（0.5），且不低于 0.75
    assert summary["gate"]["balanced_accuracy"] >= 0.75


async def test_extract_compare_closed_set_arm(db):
    from scripts.eval_memory import run_extract_compare

    out = await run_extract_compare._run(["closed_set"])
    arm = out["closed_set"]
    if arm.get("topic_accuracy") is None:
        pytest.skip("embedder 不可用（无网络/key），离线跳过")
    assert 0.0 <= arm["topic_accuracy"] <= 1.0
    assert arm["hit"] + len(arm["details"]) <= arm["total"]
