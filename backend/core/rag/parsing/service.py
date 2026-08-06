"""ParseService：缓存感知、引擎可插buff的解析入口。

查缓存→命中返回；未命中→readiness gate→reserve→引擎 parse→load_ir→write_manifest。
**格式不支持直接报错不换引擎**（单引擎哲学，低质量兜底比失败更糟）。借鉴 DeepTutor
``service.py``，cache_root 取自 settings.paths.parse_cache_dir。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from settings import get_settings

from core.rag.parsing import cache
from core.rag.parsing.registry import get_engine
from core.rag.parsing.types import ParsedDocument, ParserError

logger = logging.getLogger(__name__)


def _cache_root() -> Path:
    return Path(get_settings().paths.parse_cache_dir)


def parse_document(
    source_path: str | Path,
    *,
    engine: Optional[str] = None,
    on_output: Optional[Callable[[str], None]] = None,
) -> ParsedDocument:
    """解析文档（查缓存→调引擎），返回 ParsedDocument。

    同步阻塞：MinerU API 轮询是网络 IO 等待，在 worker 线程跑（ingestion 用
    ``asyncio.to_thread`` 包 parse_files）。失败抛 ``ParserError``（含引擎名+原因），
    调用方写 KBFile.status=error + error_msg 给前端展示。

    调度顺序（DeepTutor 不变量）：
    1. 文件存在检查
    2. 格式支持检查（不支持直接 ParserError，**不换引擎**）
    3. 算缓存键 (source_hash, signature)
    4. 查缓存命中 → load_ir 组装返回（缓存即 ready 证据，不查 readiness）
    5. readiness gate（reserve 前查，避免无谓建目录）
    6. reserve → parse → load_ir → 空产物检查 → write_manifest（最后写=ready）
    7. 异常 → cleanup_failed（清未完成目录）→ raise
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise ParserError(f"待解析文件不存在: {source_path}")

    eng = get_engine(engine)
    config = eng.resolve_config()
    engine_name = eng.name

    suffix = source_path.suffix.lower()
    supported = eng.supported_formats()
    if supported and suffix not in supported:
        raise ParserError(
            f"解析引擎 '{engine_name}' 不支持 {suffix or '该'} 格式文件（{source_path.name}）"
        )

    sig = eng.signature(config).hash()
    source_hash = cache.source_hash_from_path(source_path)
    root = _cache_root()

    # 缓存命中（不查 readiness：缓存目录存在本身就是 ready 证据）
    hit = cache.lookup(root, source_hash, sig)
    if hit is not None:
        logger.info("解析缓存命中 %s（%s/%s）", source_path.name, engine_name, sig)
        markdown, blocks, asset_dir = cache.load_ir(hit)
        return ParsedDocument(
            markdown=markdown,
            blocks=blocks,
            asset_dir=asset_dir,
            source_hash=source_hash,
            parser_signature=sig,
            engine=engine_name,
            workdir=hit,
        )

    # readiness gate（reserve 前查）
    ready, reason = eng.is_ready(config)
    if not ready:
        raise ParserError(reason or f"引擎 '{engine_name}' 未就绪")

    workdir = cache.reserve(root, source_hash, sig)
    logger.info("解析 %s（引擎 %s，签名 %s）", source_path.name, engine_name, sig)
    try:
        eng.parse(source_path, workdir, config=config, on_output=on_output)
        markdown, blocks, asset_dir = cache.load_ir(workdir)
        if not markdown and not blocks:
            raise ParserError(f"引擎 '{engine_name}' 对 {source_path.name} 未产出内容")
        cache.write_manifest(
            workdir,
            {
                "engine": engine_name,
                "signature": sig,
                "source_hash": source_hash,
                "source_name": source_path.name,
            },
        )
        return ParsedDocument(
            markdown=markdown,
            blocks=blocks,
            asset_dir=asset_dir,
            source_hash=source_hash,
            parser_signature=sig,
            engine=engine_name,
            workdir=workdir,
        )
    except Exception:
        cache.cleanup_failed(workdir)
        raise


__all__ = ["parse_document"]
