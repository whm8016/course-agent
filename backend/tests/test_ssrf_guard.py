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
