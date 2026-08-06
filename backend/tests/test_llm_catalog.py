"""catalog profile 读写单测：CRUD + get_profile + 公开/管理员视图投影 + active。

per-request provider 切换：catalog 以 profile 池形式存，
public 视图去 key（对话下拉用），admin 视图含 key（编辑回填）。
"""
import pytest

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


# ── 运行期缓存层（load_catalog_cached / get_profile_cached / invalidate）──────────
# 用 in-memory dict 替换 Redis cache_*，确保断言确定性（不依赖真实 Redis 可用性）。


@pytest.fixture
def mem_cache(monkeypatch):
    """把 catalog 引用的 cache_get/set/delete 换成 in-memory dict（确定性）。"""
    store: dict = {}

    async def fake_get(key):
        return store.get(key)

    async def fake_set(key, value, ttl=60):
        store[key] = value

    async def fake_delete(key):
        store.pop(key, None)

    monkeypatch.setattr(cat, "cache_get", fake_get)
    monkeypatch.setattr(cat, "cache_set", fake_set)
    monkeypatch.setattr(cat, "cache_delete", fake_delete)
    return store


async def test_load_catalog_cached_miss_hit_invalidate(tmp_path, monkeypatch, mem_cache):
    """miss→读文件+回填；hit→返回缓存（不重读）；invalidate→miss→重读新文件。"""
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("p1", {"binding": "openai", "text_model": "gpt-4o"})

    data1 = await cat.load_catalog_cached()  # miss → 读文件 + 回填缓存
    assert [p["id"] for p in data1["profiles"]] == ["p1"]
    assert mem_cache  # 缓存已回填

    # 文件已变（追加 p2），但缓存未失效 → 仍只看到 p1（证明命中缓存而非重读）
    cat.upsert_profile("p2", {"binding": "deepseek", "text_model": "deepseek-chat"})
    data2 = await cat.load_catalog_cached()
    assert [p["id"] for p in data2["profiles"]] == ["p1"]

    # 失效后 → miss → 重读 → 看到 p1 + p2
    await cat.invalidate_catalog_cache()
    assert not mem_cache
    data3 = await cat.load_catalog_cached()
    assert {p["id"] for p in data3["profiles"]} == {"p1", "p2"}


async def test_get_profile_cached(tmp_path, monkeypatch, mem_cache):
    """get_profile_cached 按 id 取（空/不存在返回 None）。"""
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("p1", {"binding": "openai", "text_model": "gpt-4o"})

    prof = await cat.get_profile_cached("p1")
    assert prof is not None and prof["binding"] == "openai"
    assert await cat.get_profile_cached("nope") is None
    assert await cat.get_profile_cached("") is None


async def test_active_profile_id_cached(tmp_path, monkeypatch, mem_cache):
    monkeypatch.setattr(cat, "CATALOG_PATH", str(tmp_path / "cat.json"))
    cat.upsert_profile("a", {"binding": "openai", "text_model": "gpt-4o"})
    cat.upsert_profile("b", {"binding": "deepseek", "text_model": "deepseek-chat"})
    cat.set_active("b")
    assert await cat.active_profile_id_cached() == "b"
