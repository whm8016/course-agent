"""SkillService 单测：frontmatter 解析 / 三层查找(personal>course>builtin) / name 正则 /
路径穿越拒绝 / 截断 / manifest 渲染去重+always排除 / always 注入 / builtin skill 可读。
"""
from pathlib import Path

import pytest

from core.skills.skill_service import (
    InvalidSkillNameError,
    InvalidSkillPathError,
    SkillNotFoundError,
    SkillService,
    SkillSummaryEntry,
    render_skills_manifest,
)


def _write_skill(root: Path, name: str, body: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


def test_builtin_skills_shipped():
    """项目自带的 builtin skill 能被读到。"""
    from core.skills.skill_service import BUILTIN_SKILLS_ROOT
    svc = SkillService(user_root=Path("__nonexistent_user__"), builtin_root=BUILTIN_SKILLS_ROOT)
    names = {i.name for i in svc.list_skills()}
    assert "concept-explain" in names
    assert "error-analysis" in names


def test_user_overrides_builtin(tmp_path):
    _write_skill(tmp_path / "builtin", "demo",
                 "---\nname: demo\ndescription: builtin 版\n---\n内置")
    _write_skill(tmp_path / "user", "demo",
                 "---\nname: demo\ndescription: user 版\n---\n用户")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    infos = {i.name: i for i in svc.list_skills()}
    assert infos["demo"].source == "course"
    assert infos["demo"].description == "user 版"


def test_personal_overrides_course(tmp_path):
    """personal 层覆盖 course 层，source 标记为 personal；read 也命中 personal。"""
    _write_skill(tmp_path / "course", "demo",
                 "---\nname: demo\ndescription: 课程版\n---\n课程内容")
    _write_skill(tmp_path / "personal", "demo",
                 "---\nname: demo\ndescription: 个人版\n---\n个人内容")
    svc = SkillService(
        user_root=tmp_path / "course",
        builtin_root=None,
        personal_root=tmp_path / "personal",
    )
    infos = {i.name: i for i in svc.list_skills()}
    assert infos["demo"].source == "personal"
    assert infos["demo"].description == "个人版"
    assert "个人内容" in svc.read_skill_file("demo")


def test_read_skill_file_returns_full(tmp_path):
    _write_skill(tmp_path / "builtin", "demo",
                 "---\nname: demo\ndescription: d\n---\n正文内容")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    content = svc.read_skill_file("demo")
    assert "正文内容" in content


def test_read_skill_references_file(tmp_path):
    _write_skill(tmp_path / "builtin", "demo",
                 "---\nname: demo\ndescription: d\n---\nbody")
    (tmp_path / "builtin" / "demo" / "references").mkdir()
    (tmp_path / "builtin" / "demo" / "references" / "extra.md").write_text("更多细节", encoding="utf-8")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    assert svc.read_skill_file("demo", "references/extra.md") == "更多细节"


def test_path_traversal_rejected(tmp_path):
    _write_skill(tmp_path / "builtin", "demo",
                 "---\nname: demo\ndescription: d\n---\nbody")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    with pytest.raises(InvalidSkillPathError):
        svc.read_skill_file("demo", "../../etc/passwd")
    with pytest.raises(InvalidSkillPathError):
        svc.read_skill_file("demo", "/etc/passwd")  # 绝对路径


def test_invalid_name_rejected(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    with pytest.raises(InvalidSkillNameError):
        svc.read_skill_file("Bad Name!")


def test_not_found(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    with pytest.raises(SkillNotFoundError):
        svc.read_skill_file("ghost")


def test_truncation(tmp_path, monkeypatch):
    import core.skills.skill_service as ss
    monkeypatch.setattr(ss, "_MAX_READ_CHARS", 10)
    _write_skill(tmp_path / "builtin", "big",
                 "---\nname: big\ndescription: d\n---\n" + "x" * 100)
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    assert "[... truncated ...]" in svc.read_skill_file("big")


def test_render_manifest_dedup_and_exclude_always():
    entries = [
        SkillSummaryEntry(name="a", description="技能 A"),
        SkillSummaryEntry(name="a", description="重复"),            # 去重
        SkillSummaryEntry(name="b", description="", always=True),   # always 排除
        SkillSummaryEntry(name="c", description="技能 C", available=False, missing=["ENV: X"]),
    ]
    out = render_skills_manifest(entries)
    assert "- **a** — 技能 A" in out
    assert "重复" not in out
    assert "- **b**" not in out
    assert "- **c** — 技能 C (unavailable: ENV: X)" in out
    assert out.startswith("## Skills")


def test_render_manifest_empty():
    assert render_skills_manifest([]) == ""


def test_always_injection(tmp_path):
    _write_skill(tmp_path / "builtin", "house",
                 "---\nname: house\ndescription: 守则\nalways: true\n---\n每轮都要遵守")
    _write_skill(tmp_path / "builtin", "ondemand",
                 "---\nname: ondemand\ndescription: 按需\n---\n按需读")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    always = svc.load_always_for_context()
    assert "每轮都要遵守" in always
    assert "按需读" not in always
    manifest = render_skills_manifest(svc.summary_entries())
    assert "- **house**" not in manifest   # always 不进 manifest
    assert "- **ondemand**" in manifest


def test_render_manifest_truncates_long_description():
    """单条 description 超长 → 截断到 _MANIFEST_DESC_MAX_CHARS 并收省略号（对齐官方 ~100 词/skill）。"""
    entries = [SkillSummaryEntry(name="big", description="字" * 500)]
    out = render_skills_manifest(entries)
    assert "字" * 500 not in out      # 未整段塞入
    assert "…" in out                 # 带省略号
    assert "- **big**" in out         # 条目仍在


def test_render_manifest_entry_cap_drops_tail(monkeypatch):
    """条数超上限：按 personal>course>builtin 优先级从尾部裁剪并计数。"""
    import core.skills.skill_service as ss
    monkeypatch.setattr(ss, "_MANIFEST_MAX_ENTRIES", 3)
    monkeypatch.setattr(ss, "_MANIFEST_MAX_CHARS", 100_000)  # 让条数成为唯一约束
    entries = [SkillSummaryEntry(name=f"s{i}", description=f"desc{i}") for i in range(5)]
    out = render_skills_manifest(entries)
    assert "- **s0**" in out and "- **s2**" in out          # 前 3 条保留
    assert "- **s3**" not in out and "- **s4**" not in out  # 尾部 2 条裁掉
    assert "另有 2 个" in out                                # 省略计数准确


def test_render_manifest_char_budget_drops_tail(monkeypatch):
    """字符预算触发（条数未超）：从尾部移除并计数，首条（最高优先级）必保留。"""
    import core.skills.skill_service as ss
    monkeypatch.setattr(ss, "_MANIFEST_MAX_ENTRIES", 100)
    monkeypatch.setattr(ss, "_MANIFEST_MAX_CHARS", 200)     # 极小预算，触发字符裁剪
    entries = [SkillSummaryEntry(name=f"x{i}", description="abcdefghij") for i in range(10)]
    out = render_skills_manifest(entries)
    assert "- **x0**" in out        # 首条必保留
    assert "- **x9**" not in out    # 尾条必被裁
    assert "另有" in out            # 必然有省略计数
