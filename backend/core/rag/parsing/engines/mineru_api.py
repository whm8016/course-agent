"""MinerU 托管 API 解析引擎（同步 httpx，worker 线程跑避免嵌套 loop）。

四步流程（https://mineru.net/apiManage/docs）：
1. POST /api/v4/file-urls/batch → {batch_id, file_urls:[signed_upload_url]}
2. PUT 原始字节到 signed_upload_url（不带 Content-Type/Auth，否则破坏 OSS 签名）
3. 轮询 GET /api/v4/extract-results/batch/{batch_id} → state=done/failed + full_zip_url
4. 下载 full_zip_url 解包到 workdir（full.md + content_list.json + images/）

业务码双层检查（HTTP 200 里 code!=0 也错）。zip Slip/bomb 防护。借鉴 DeepTutor
``engines/mineru/cloud.py``。唯一默认引擎——云端不装 torch，worker 内存 ~2.5GB→~0.4GB。
"""
from __future__ import annotations

import io
import logging
import ssl
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from core.rag.parsing.signature import ParserSignature
from core.rag.parsing.types import ParserError
from settings import get_settings

logger = logging.getLogger(__name__)

_SUBMIT_TIMEOUT = 60.0
_UPLOAD_TIMEOUT = 300.0
_DOWNLOAD_TIMEOUT = 300.0
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF_SEC = 3.0
_MAX_TOTAL_BYTES = 500 * 1024 * 1024

# Python 3.10~3.12 + OpenSSL 3 对"对端未发 close_notify 就断连"判定过严，会把这类
# 数据实际已收完、只是连接没优雅关闭的情况报成 SSLEOFError（3.9/3.13 不受影响，
# 见 https://github.com/urllib3/urllib3/issues/2733）。MinerU 结果 zip 走的 CDN
# （cdn-mineru.openxlab.org.cn）恰好是这类服务器，每次下载 100% 复现。
# 只对下载这一步放宽该检查：截断攻击风险可接受（zip_url 是一次性签名地址，非用户输入，
# 且已有 zip 完整性校验 + 大小上限兜底）。
_DOWNLOAD_SSL_CONTEXT = ssl.create_default_context()
if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
    _DOWNLOAD_SSL_CONTEXT.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
_MAX_ENTRIES = 5000
_TERMINAL_OK = "done"
_TERMINAL_FAIL = "failed"


class MinerUError(ParserError):
    """MinerU 解析失败（token/限流/超时/坏 zip/超页超限）。"""


class MinerUApiEngine:
    """MinerU 托管 API 解析（PDF → markdown + content_list）。"""

    name = "mineru_api"

    @classmethod
    def is_available(cls) -> bool:
        # 托管 API 无 Python 重依赖（只需 httpx，已在主依赖），始终"可导入"；
        # 能否真跑由 is_ready 探测 api_key。
        return True

    def resolve_config(self) -> dict[str, Any]:
        cfg = get_settings().parsing
        return {
            "api_base_url": cfg.mineru_base_url,
            "api_token": cfg.mineru_api_key.get_secret_value(),
            "model_version": cfg.mineru_model,
            "language": cfg.mineru_language,
            "enable_formula": bool(cfg.enable_formula),
            "enable_table": bool(cfg.enable_table),
            "poll_interval": cfg.poll_interval,
            "poll_timeout": cfg.poll_timeout,
            "max_file_mb": cfg.max_file_mb,
        }

    def supported_formats(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def signature(self, config: dict[str, Any]) -> ParserSignature:
        # 只折叠影响输出的旋钮；api_token 永不进 signature（换 token 不该让缓存失效）
        return ParserSignature.build(
            "mineru_api",
            f"cloud:{config['api_base_url']}",
            {
                "model_version": config["model_version"],
                "language": config["language"],
                "enable_formula": config["enable_formula"],
                "enable_table": config["enable_table"],
            },
        )

    def is_ready(self, config: dict[str, Any]) -> tuple[bool, str]:
        if not (config.get("api_token") or "").strip():
            return False, "MinerU API token 未配置（设 PARSING__MINERU_API_KEY）"
        return True, ""

    def parse(
        self,
        source_path: Path,
        workdir: Path,
        *,
        config: dict[str, Any],
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        """四步流程解析 PDF，产物写 workdir（full.md + content_list.json + images/）。"""
        pdf_path = Path(source_path)
        if not pdf_path.is_file():
            raise MinerUError(f"PDF 文件不存在: {pdf_path}")
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > config["max_file_mb"]:
            raise MinerUError(
                f"文件 {pdf_path.name} {size_mb:.1f}MB 超过 MinerU 上限 {config['max_file_mb']}MB"
            )

        base_url = str(config["api_base_url"]).rstrip("/")
        headers = {"Authorization": f"Bearer {config['api_token']}", "Accept": "application/json"}

        def report(msg: str) -> None:
            if on_output:
                try:
                    on_output(msg)
                except Exception:
                    logger.debug("on_output 回调失败", exc_info=True)

        with httpx.Client(base_url=base_url, headers=headers) as client:
            report(f"MinerU: 申请上传槽 {pdf_path.name}（{size_mb:.1f}MB）")
            batch_id, upload_url = _request_upload(client, pdf_path, config)
            report(f"MinerU: 上传 {pdf_path.name}")
            _upload_file(pdf_path, upload_url)
            report("MinerU: 等待解析（轮询）")
            zip_url = _poll_for_zip(
                client,
                batch_id,
                pdf_path.name,
                poll_interval=config["poll_interval"],
                timeout=config["poll_timeout"],
                on_progress=on_output,
            )
            report("MinerU: 下载结果")
            archive = _download(zip_url)

        report("MinerU: 解包")
        _extract_archive(archive, workdir)
        logger.info("MinerU 解析完成: %s → %s", pdf_path.name, workdir)


# ── 四步流程 ──────────────────────────────────────────────────────────────────


def _request_upload(
    client: httpx.Client, pdf_path: Path, config: dict[str, Any]
) -> tuple[str, str]:
    body: dict[str, Any] = {
        "files": [{"name": pdf_path.name, "is_ocr": False}],
        "model_version": config["model_version"],
        "enable_formula": config["enable_formula"],
        "enable_table": config["enable_table"],
    }
    lang = config.get("language")
    if lang and lang != "auto":
        body["language"] = lang
    payload = _post_json(client, "/api/v4/file-urls/batch", body)
    data = payload.get("data") or {}
    batch_id = str(data.get("batch_id") or "").strip()
    file_urls = data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise MinerUError("MinerU 未返回上传 URL（batch_id/file_urls 缺失）")
    return batch_id, str(file_urls[0])


def _upload_file(pdf_path: Path, upload_url: str) -> None:
    # signed URL 自带鉴权；不能带 Authorization/Content-Type（多余 header 破坏 OSS 签名）
    try:
        resp = httpx.put(upload_url, content=pdf_path.read_bytes(), timeout=_UPLOAD_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise MinerUError(f"上传 PDF 到 MinerU 失败: {exc}") from exc


def _poll_for_zip(
    client: httpx.Client,
    batch_id: str,
    file_name: str,
    *,
    poll_interval: int,
    timeout: int,
    on_progress: Optional[Callable[[str], None]] = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_state = ""
    last_report = ""
    while True:
        payload = _get_json(client, f"/api/v4/extract-results/batch/{batch_id}")
        results = (payload.get("data") or {}).get("extract_result") or []
        entry = _match_entry(results, file_name)
        if entry is not None:
            state = str(entry.get("state") or "").strip().lower()
            last_state = state or last_state
            if on_progress is not None:
                prog = entry.get("extract_progress") or {}
                msg = f"MinerU: {state or '排队中'}"
                if prog.get("total_pages"):
                    msg += f"（{prog.get('extracted_pages') or 0}/{prog['total_pages']} 页）"
                if msg != last_report:
                    last_report = msg
                    try:
                        on_progress(msg)
                    except Exception:
                        on_progress = None
            if state == _TERMINAL_OK:
                zip_url = str(entry.get("full_zip_url") or "").strip()
                if not zip_url:
                    raise MinerUError("MinerU 报 done 但无 full_zip_url")
                return zip_url
            if state == _TERMINAL_FAIL:
                err = str(entry.get("err_msg") or "未知错误")
                raise MinerUError(f"MinerU 解析失败: {err}")
        if time.monotonic() >= deadline:
            raise MinerUError(
                f"MinerU 解析超时（{timeout}s，最后状态 {last_state or '未知'}）"
            )
        time.sleep(poll_interval)


def _download(zip_url: str) -> bytes:
    # CDN 连接在传完前就被中途掐断（非"忘发 close_notify"这种误判，是真截断），
    # 每次都在类似位置栽，从头重下大概率再栽一次。改成断点续传：失败后用 Range
    # 从已收字节数继续要，单次连接只需扛住"剩余那一段"，重试更容易拼出完整文件。
    buf = bytearray()
    last_exc: httpx.HTTPError | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        headers = {"Range": f"bytes={len(buf)}-"} if buf else {}
        try:
            with httpx.stream(
                "GET",
                zip_url,
                timeout=_DOWNLOAD_TIMEOUT,
                follow_redirects=True,
                verify=_DOWNLOAD_SSL_CONTEXT,
                headers=headers,
            ) as resp:
                if buf and resp.status_code == 416:
                    # 服务器认为没有更多字节可给，说明上次已经收完了
                    break
                resp.raise_for_status()
                if buf and resp.status_code != 206:
                    # 服务器不支持 Range（忽略了该 header，返回完整 200），只能从头收
                    logger.warning("MinerU CDN 不支持断点续传，重新完整下载")
                    buf.clear()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
            return bytes(buf)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < _DOWNLOAD_RETRIES:
                wait = _DOWNLOAD_RETRY_BACKOFF_SEC * attempt
                logger.warning(
                    "下载 MinerU 结果失败（第 %d/%d 次，已收 %d 字节）: %s，%.0fs 后重试",
                    attempt, _DOWNLOAD_RETRIES, len(buf), exc, wait,
                )
                time.sleep(wait)
    raise MinerUError(f"下载 MinerU 结果失败（重试 {_DOWNLOAD_RETRIES} 次仍失败）: {last_exc}") from last_exc


# ── helpers ───────────────────────────────────────────────────────────────────


def _match_entry(results: list, file_name: str) -> Optional[dict]:
    rows = [r for r in results if isinstance(r, dict)]
    if not rows:
        return None
    for row in rows:
        if str(row.get("file_name") or "") == file_name:
            return row
    return rows[0]


def _post_json(client: httpx.Client, path: str, body: dict) -> dict:
    try:
        resp = client.post(path, json=body, timeout=_SUBMIT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as exc:
        raise MinerUError(_http_error_message(exc)) from exc
    except httpx.HTTPError as exc:
        raise MinerUError(f"MinerU API 请求失败: {exc}") from exc
    _check_code(payload)
    return payload


def _get_json(client: httpx.Client, path: str) -> dict:
    try:
        resp = client.get(path, timeout=_SUBMIT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as exc:
        raise MinerUError(_http_error_message(exc)) from exc
    except httpx.HTTPError as exc:
        raise MinerUError(f"MinerU API 请求失败: {exc}") from exc
    _check_code(payload)
    return payload


def _check_code(payload: dict) -> None:
    # MinerU 在 HTTP 200 里也包 {"code": <非0>, "msg": ...}，必须查业务 code
    if not isinstance(payload, dict):
        raise MinerUError("MinerU API 返回非 JSON")
    code = payload.get("code")
    if code not in (0, None):
        msg = str(payload.get("msg") or "未知错误")
        raise MinerUError(f"MinerU API 错误（code {code}）: {msg}")


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    if status in (401, 403):
        return "MinerU API token 被拒（401/403），检查 PARSING__MINERU_API_KEY"
    if status == 429:
        return "MinerU API 限流（429），稍后重试或降低并发"
    return f"MinerU API 返回 HTTP {status}"


def _extract_archive(archive_bytes: bytes, target_dir: Path) -> None:
    """解包 MinerU zip 到 target_dir，保留 images/ 子目录，防 Zip Slip/bomb。

    不做扩展名白名单（信任 MinerU 产物，非用户上传）。
    """
    target_root = target_dir.resolve()
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) > _MAX_ENTRIES:
                raise MinerUError(f"MinerU zip 条目过多（{len(members)}）")
            for member in members:
                rel = Path(member.filename.replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    logger.warning("跳过不安全 zip 条目: %s", member.filename)
                    continue
                dest = (target_root / rel).resolve()
                if target_root not in dest.parents and dest != target_root:
                    logger.warning("跳过越界 zip 条目: %s", member.filename)
                    continue
                total += member.file_size
                if total > _MAX_TOTAL_BYTES:
                    raise MinerUError("MinerU zip 超过大小限制")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
    except zipfile.BadZipFile as exc:
        raise MinerUError(f"MinerU 返回无效 zip: {exc}") from exc


__all__ = ["MinerUApiEngine", "MinerUError"]
