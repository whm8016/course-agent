"""SkillService 写入 CRUD 单测：create/update/delete + frontmatter 管理 +
builtin 只读保护 + 重名校验 + rename + get_detail。
"""
import pytest

from core.skills.skill_service import (
    InvalidSkillNameError,
    SkillExistsError,
    SkillNotFoundError,
    SkillReadOnlyError,
    SkillService,
)


def _write_builtin(root, name, body):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


def test_create_user_skill(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    info = svc.create("my-skill", "测试技能", "做某事的步骤", always=True)
    assert info.name == "my-skill"
    assert info.source in ("course", "personal")  # user 创建的 skill（course 或 personal 层）
    text = (tmp_path / "user" / "my-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: my-skill" in text
    assert "always: true" in text
    assert "做某事的步骤" in text


def test_create_duplicate_rejected(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    svc.create("dup", "d", "body")
    with pytest.raises(SkillExistsError):
        svc.create("dup", "d2", "body2")


def test_create_invalid_name(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    with pytest.raises(InvalidSkillNameError):
        svc.create("Bad Name", "d", "body")


def test_update_description_and_content(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    svc.create("s1", "旧描述", "旧正文")
    info = svc.update("s1", description="新描述", content="新正文", always=True)
    assert info.description == "新描述"
    detail = svc.get_detail("s1")
    assert detail["description"] == "新描述"
    assert detail["always"] is True
    assert "新正文" in detail["content"]


def test_update_rename(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    svc.create("old-name", "d", "body")
    svc.update("old-name", rename_to="new-name")
    assert svc.get_detail("new-name")["name"] == "new-name"
    with pytest.raises(SkillNotFoundError):
        svc.get_detail("old-name")


def test_delete_user_skill(tmp_path):
    svc = SkillService(user_root=tmp_path / "user", builtin_root=tmp_path / "builtin")
    svc.create("todel", "d", "body")
    svc.delete("todel")
    with pytest.raises(SkillNotFoundError):
        svc.get_detail("todel")


def test_builtin_readonly(tmp_path):
    bi = tmp_path / "builtin"
    _write_builtin(bi, "builtin-skill", "---\nname: builtin-skill\ndescription: d\n---\nbody")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=bi)
    with pytest.raises(SkillReadOnlyError):
        svc.update("builtin-skill", description="hack")
    with pytest.raises(SkillReadOnlyError):
        svc.delete("builtin-skill")


def test_get_detail_returns_body_and_meta(tmp_path):
    bi = tmp_path / "builtin"
    _write_builtin(bi, "demo", "---\nname: demo\ndescription: 演示\nalways: true\n---\n正文内容")
    svc = SkillService(user_root=tmp_path / "user", builtin_root=bi)
    detail = svc.get_detail("demo")
    assert detail["name"] == "demo"
    assert detail["description"] == "演示"
    assert detail["always"] is True
    assert detail["content"] == "正文内容"  # body 不含 frontmatter
    assert detail["source"] == "builtin"
    assert detail["read_only"] is True
