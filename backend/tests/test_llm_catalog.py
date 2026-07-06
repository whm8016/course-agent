"""catalog profile 读写单测：CRUD + get_profile + 公开/管理员视图投影 + active。

per-request provider 切换：catalog 以 profile 池形式存，
public 视图去 key（对话下拉用），admin 视图含 key（编辑回填）。
"""
from core.llm import catalog as cat


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "nope.json"))
    data = cat.load_catalog()
    assert data["active_profile"] == "default"
    assert data["profiles"] == []


def test_upsert_get_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("p1", {
        "name": "P1", "binding": "deepseek", "text_model": "deepseek-chat", "api_key": "sk-x",
    })
    prof = cat.get_profile("p1")
    assert prof is not None
    assert prof["binding"] == "deepseek"
    assert prof["models"]["text"]["model"] == "deepseek-chat"
    assert cat.profile_text_model(prof) == "deepseek-chat"

    # public view 去掉 key
    pub = cat.profile_public_view(prof, "p1")
    assert "api_key" not in pub
    assert pub["active"] is True
    assert pub["text_model"] == "deepseek-chat"

    # admin view 含 key（编辑回填）
    adm = cat.profile_admin_view(prof, "p1")
    assert adm["api_key"] == "sk-x"

    # 不存在
    assert cat.get_profile("nope") is None
    # delete
    assert cat.delete_profile("p1") is True
    assert cat.get_profile("p1") is None


def test_upsert_updates_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("p1", {"binding": "openai", "text_model": "gpt-4o"})
    assert len(cat.list_profiles()) == 1
    # 同 id 再次 upsert = 更新而非新增
    cat.upsert_profile("p1", {"binding": "openai", "text_model": "gpt-4o-mini"})
    assert len(cat.list_profiles()) == 1
    assert cat.profile_text_model(cat.get_profile("p1")) == "gpt-4o-mini"


def test_set_active(tmp_path, monkeypatch):
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("a", {"binding": "openai", "text_model": "gpt-4o"})
    cat.upsert_profile("b", {"binding": "deepseek", "text_model": "deepseek-chat"})
    assert cat.set_active("b") is True
    assert cat.active_profile_id() == "b"
    assert cat.set_active("nope") is False


def test_delete_active_falls_back(tmp_path, monkeypatch):
    """删除当前 active profile 时，active 回退到剩余第一个。"""
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("a", {"binding": "openai", "text_model": "gpt-4o"})
    cat.upsert_profile("b", {"binding": "deepseek", "text_model": "deepseek-chat"})
    cat.set_active("a")
    cat.delete_profile("a")
    assert cat.active_profile_id() == "b"
