"""P1 SSRF 防护回归（utils/url_guard.assert_public_http_url）。

攻击者视角：普通用户在探测端点（/search_config/probe、/llm/me/test）填内网/元数据
base_url，让后端代为 GET 内网服务或云元数据端点。修复后 assert_public_http_url 解析
host 的全部 IP，任一落入私网/环回/链路本地/保留段即拒。
"""
import pytest

from utils.url_guard import assert_public_http_url


def test_rejects_loopback():
    with pytest.raises(ValueError):
        assert_public_http_url("http://127.0.0.1:8080")


# ── 端点收口：guard 此前只护 probe 端点，不护「保存后运行时」路径 ────────────
# 攻击链：学生 PUT 私网 base_url/proxy 落库（200）→ 对话中触发 web_search/自配
# LLM/视觉 → 后端替学生访问内网/元数据端点，响应经 LLM 回显。


async def _student_token(client, username: str) -> str:
    reg = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "pass1234", "display_name": "S"},
    )
    assert reg.status_code == 200
    return reg.json()["token"]


async def test_put_llm_me_rejects_private_base_url(client):
    """PUT /api/llm/me 存元数据端点/环回 base_url → 400 拒绝（修复前 200 落库）。"""
    token = await _student_token(client, "ssrf_llm_1")
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.put(
        "/api/llm/me",
        headers=headers,
        json={"binding": "openai", "api_key": "sk-x",
              "base_url": "http://169.254.169.254/latest/meta-data"},
    )
    assert r.status_code == 400
    r2 = await client.put(
        "/api/llm/me",
        headers=headers,
        json={"binding": "openai", "api_key": "sk-x",
              "vision_base_url": "http://127.0.0.1:9000"},
    )
    assert r2.status_code == 400


async def test_put_llm_me_accepts_public_base_url(client):
    """公网 base_url 正常保存（守门不误伤合法自配）。"""
    token = await _student_token(client, "ssrf_llm_2")
    r = await client.put(
        "/api/llm/me",
        headers={"Authorization": f"Bearer {token}"},
        json={"binding": "openai", "api_key": "sk-x", "base_url": "http://8.8.8.8/v1"},
    )
    assert r.status_code == 200


async def test_put_search_config_me_rejects_private_base_url_and_proxy(client):
    token = await _student_token(client, "ssrf_search_1")
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.put(
        "/api/search_config/me",
        headers=headers,
        json={"provider": "searxng", "base_url": "http://10.0.0.5:8080"},
    )
    assert r.status_code == 400
    r2 = await client.put(
        "/api/search_config/me",
        headers=headers,
        json={"provider": "brave", "api_key": "k",
              "base_url": "http://8.8.8.8", "proxy": "http://192.168.1.9:7890"},
    )
    assert r2.status_code == 400


async def test_active_provider_view_drops_legacy_private_row(db):
    """存量脏行（收口前已落库的私网 base_url）读取侧被拒：返回 None 回落平台默认。

    持久化校验挡新写入；读取侧校验兜 DNS rebinding（保存时公网、调用时解析到
    私网）与修复前已落库的行。
    """
    from core.db.database import AsyncSessionLocal, User, UserLLMProvider

    async with AsyncSessionLocal() as s:
        s.add(User(id="u_ssrf", username="stu_ssrf", password_hash="x", role="student"))
        s.add(UserLLMProvider(user_id="u_ssrf", binding="openai",
                              api_key_encrypted="enc", base_url="http://127.0.0.1:9000"))
        await s.commit()

    from core.db.user_llm_provider import get_active_provider_view

    assert await get_active_provider_view("u_ssrf") is None


def test_resolve_search_user_private_url_dropped(tmp_path, monkeypatch):
    """用户 override 的私网 base_url/proxy 视同未配置，回落 admin/env（不抛错）。"""
    monkeypatch.setenv("SEARCH_PROVIDER", "")
    monkeypatch.setenv("SEARCH_BASE_URL", "")
    monkeypatch.setenv("SEARCH_PROXY", "")
    from services.search.config import resolve_search_config

    cfg = resolve_search_config(
        user_override={"provider": "searxng", "base_url": "http://127.0.0.1:8888",
                       "api_key": "k", "proxy": "http://10.1.2.3:7890"}
    )
    # searxng 失去 base_url 后走既有回落链 → duckduckgo；用户的私网 proxy 被丢
    # （settings/env 层的代理属部署者信任域，不在本测试断言范围）
    assert cfg.provider == "duckduckgo"
    assert cfg.proxy != "http://10.1.2.3:7890"


def test_rejects_rfc1918_private():
    with pytest.raises(ValueError):
        assert_public_http_url("http://10.0.0.1")
    with pytest.raises(ValueError):
        assert_public_http_url("http://192.168.1.1")
    with pytest.raises(ValueError):
        assert_public_http_url("http://172.16.0.1")


def test_rejects_cloud_metadata_endpoint():
    """169.254.169.254（AWS/Azure/GCP 元数据）属 link-local，必须拒。"""
    with pytest.raises(ValueError):
        assert_public_http_url("http://169.254.169.254/latest/meta-data/")


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        assert_public_http_url("ftp://example.com")
    with pytest.raises(ValueError):
        assert_public_http_url("file:///etc/passwd")


def test_rejects_empty():
    with pytest.raises(ValueError):
        assert_public_http_url("")


def test_allows_public_ip():
    """公网 IP（1.1.1.1）不抛--合法的自定义 endpoint 应放行。"""
    assert_public_http_url("https://1.1.1.1")  # 不抛即通过
