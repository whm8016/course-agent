"""SSRF 防护：拒绝私网 / 环回 / 链路本地 / 云元数据 URL。

供用户可达的探测端点（search_config/probe、llm/me/test）在发起外部请求前校验
base_url，防止普通用户把后端当跳板访问内网服务或云元数据端点（169.254.169.254）。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def assert_public_http_url(url: str) -> None:
    """校验 URL 指向公网 HTTP(S)，拒绝 SSRF 跳板。

    解析 host -> getaddrinfo 取所有 A/AAAA 记录 -> 逐个判 IP 是否公网。任一记录落入
    私网/环回/链路本地/保留段即拒（防 DNS rebinding：一条公网一条私网时仍拒）。

    Args:
        url: 待校验的绝对 URL。

    Raises:
        ValueError: URL 非 http(s) / 缺 host / host 解析失败 / 解析到非公网 IP。
    """
    if not url or not url.strip():
        raise ValueError("URL 不能为空")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"仅允许 http/https URL，得到 {parsed.scheme or '空'}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL 缺少 host")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析 host {host}: {exc}")

    for info in infos:
        ip_str = info[4][0]
        # IPv6 getaddrinfo 可能返回 ::ffff:1.2.3.4 形式，ip_address 能正确识别
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local  # 覆盖 169.254.169.254 云元数据
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"URL 指向非公网地址 {ip_str}（拒绝 SSRF）")


def is_public_http_url(url: str) -> bool:
    """非抛错版守门（读取侧降级路径用：非公网 → False 而非异常）。"""
    try:
        assert_public_http_url(url)
    except ValueError:
        return False
    return True


def require_public_http_url_fields(obj, fields: tuple[str, ...]) -> None:
    """批量校验对象上多个 URL 字段（空值=未配置，跳过不校验）。

    写侧（PUT /llm/me、PUT /search_config/me 等）统一走这里——字段清单与错误
    文案（"{field}: {reason}"）单一真相源，api 层只需一个 try/except 映射 400。
    """
    for field in fields:
        url = str(getattr(obj, field, "") or "").strip()
        if url:
            try:
                assert_public_http_url(url)
            except ValueError as exc:
                raise ValueError(f"{field}: {exc}") from exc


__all__ = [
    "assert_public_http_url",
    "is_public_http_url",
    "require_public_http_url_fields",
]
