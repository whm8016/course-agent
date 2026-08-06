"""Phase 5: L3 procedural——巩固生成 personal SKILL.md 草稿（不自动 always + 审核位）。

覆盖：draft prompt 构建、write 不开 always 且标 auto_generated、已存在跳过、门槛、LLM 调用。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory import procedural


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


def test_build_draft_prompt_includes_weak_points_only():
    """高风险(≥0.5)知识点进 prompt；低风险排除。"""
    rows = [
        SimpleNamespace(label="导数", risk=0.8, observation_count=3),
        SimpleNamespace(label="加减法", risk=0.2, observation_count=1),
    ]
    prompt = procedural.build_draft_prompt(rows, "math101")
    assert "导数" in prompt
    assert "加减法" not in prompt
    assert "math101" in prompt


def test_draft_name_sanitizes_course_id():
    assert procedural._draft_name("Math-101!") == "profile-math-101"
    assert procedural._draft_name("") == "profile-global"


def test_write_skill_draft_not_always_with_review_flag():
    """草稿以 always=False 写 personal 层，frontmatter 标 auto_generated 待审核。"""
    svc = MagicMock()
    with patch("core.skills.skill_service.get_skill_service", return_value=svc):
        name = procedural.write_skill_draft("math101", "u1", "草稿正文")

    assert name == "profile-math101"
    svc.create.assert_called_once()
    kwargs = svc.create.call_args.kwargs
    assert kwargs["always"] is False  # 不自动 always（需人工确认）
    assert "auto_generated: true" in kwargs["content"]  # 审核位
    assert "待人工确认" in kwargs["description"]


def test_write_skill_draft_skips_if_exists():
    """同名草稿已存在 → 跳过（不覆盖），返回 None。"""
    from core.skills.skill_service import SkillExistsError

    svc = MagicMock()
    svc.create.side_effect = SkillExistsError("profile-math101")
    with patch("core.skills.skill_service.get_skill_service", return_value=svc):
        name = procedural.write_skill_draft("math101", "u1", "body")
    assert name is None


async def test_maybe_generate_below_threshold_no_draft(db):
    """累计观测数 < 阈值 → 不生成（不调 LLM）。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery

    async with AsyncSessionLocal() as s:
        s.add(KnowledgeMastery(
            user_id="u1", course_id="c1", kp_id="k1", label="x",
            mastery=0.5, risk=0.5, observation_count=5,
        ))
        await s.commit()
    with patch("core.memory.procedural.generate_skill_draft", new=AsyncMock()) as gen:
        async with AsyncSessionLocal() as s:
            ret = await procedural.maybe_generate_procedural(s, "u1", "c1")
    assert ret is None
    gen.assert_not_called()


async def test_maybe_generate_above_threshold_writes(db):
    """累计观测数 ≥ 阈值 → 生成草稿并写入。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery

    async with AsyncSessionLocal() as s:
        s.add(KnowledgeMastery(
            user_id="u1", course_id="c1", kp_id="k1", label="x",
            mastery=0.5, risk=0.5, observation_count=25,
        ))
        await s.commit()
    with patch(
        "core.memory.procedural.generate_skill_draft", new=AsyncMock(return_value="草稿正文")
    ), patch("core.memory.procedural.write_skill_draft", return_value="profile-c1") as w:
        async with AsyncSessionLocal() as s:
            ret = await procedural.maybe_generate_procedural(s, "u1", "c1")
    assert ret == "profile-c1"
    w.assert_called_once_with("c1", "u1", "草稿正文")


async def test_generate_skill_draft_calls_llm():
    """generate_skill_draft 调用 LLM 并返回正文。"""
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="学习模式总结"))]
    with patch("core.llm.llm.client") as client:
        client.chat.completions.create = AsyncMock(return_value=fake_resp)
        out = await procedural.generate_skill_draft("c1", "u1", [])
    assert out == "学习模式总结"
