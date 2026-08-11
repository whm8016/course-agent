"""内容寻址解析缓存 + IR loader。

键 = ``(source_hash, parser_signature)``，跨消费者共享（同一文件多 KB 复用、换引擎
旧结果不失效）。布局::

    parse_cache/<hash[:2]>/<source_hash>/<signature>/
        manifest.json              # 最后写 → 存在即 ready（半写目录永不命中）
        <stem>.md / full.md
        content_list.json          # 可选（产出结构的引擎才有）
        images/                    # 可选

借鉴 DeepTutor ``cache.py``，简化嵌套处理（MinerU cloud zip 直接解包到目录根）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
_READ_CHUNK = 1 << 20  # 1 MiB


def source_hash_from_path(path: Path) -> str:
    """哈希文件**字节**（非文件名），同一文档以随机临时名重传仍命中缓存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def signature_dir(cache_root: Path, source_hash: str, sig_hash: str) -> Path:
    """``<hash[:2]>`` 前缀分片，避免单目录文件过多。"""
    return cache_root / source_hash[:2] / source_hash / sig_hash


def is_ready(workdir: Optional[Path]) -> bool:
    """manifest.json 存在 == ready。半写目录（无 manifest）不算命中。"""
    return bool(workdir) and (workdir / MANIFEST_FILENAME).is_file()


def lookup(cache_root: Path, source_hash: str, sig_hash: str) -> Optional[Path]:
    """命中返回 ready 目录，否则 None。

    命中时刷新目录 mtime（对标 Bazel commit 7a774ff：缓存读命中更新 recency），否则按
    mtime LRU 裁剪的磁盘 GC（storage.gc.sweep_by_size_lru）会误删仍在用的热缓存。用
    ``os.utime`` 而非 ``Path.touch()``——后者对目录会因 ``O_WRONLY|O_CREAT`` 打开抛
    IsADirectoryError。best-effort：touch 失败（权限/只读挂载等）不影响命中返回。
    """
    target = signature_dir(cache_root, source_hash, sig_hash)
    if not is_ready(target):
        return None
    try:
        os.utime(target, None)
    except OSError:
        pass
    return target


def reserve(cache_root: Path, source_hash: str, sig_hash: str) -> Path:
    """创建/复用 signature 目录。未完成的旧目录（无 manifest，上次崩溃）先清掉。"""
    target = signature_dir(cache_root, source_hash, sig_hash)
    if target.exists() and not is_ready(target):
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_manifest(workdir: Path, meta: dict[str, Any]) -> None:
    """stamp ready。最后写，半写目录永不读成命中。"""
    payload = {
        **meta,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }
    with open(workdir / MANIFEST_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def cleanup_failed(workdir: Path) -> None:
    """best-effort 删除未完成（无 manifest）目录，不让坏目录污染下次 lookup。"""
    try:
        if workdir.is_dir() and not is_ready(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
    except Exception as exc:  # best-effort
        logger.warning("清理失败解析目录 %s 失败: %s", workdir, exc)


def load_ir(workdir: Path) -> tuple[str, Optional[list[dict]], Optional[Path]]:
    """从解析/缓存目录读 ``(markdown, blocks, asset_dir)``。

    引擎和缓存命中走同一条路径（统一 IR 组装）。markdown 找 ``*.md``（full.md 优先），
    blocks 找 ``*content_list*.json``，asset_dir 是 ``images/``（若有）。
    """
    markdown = ""
    md_files = sorted(workdir.glob("*.md"), key=lambda p: (p.name != "full.md", p.name))
    if md_files:
        try:
            markdown = md_files[0].read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("读取 markdown 失败 %s: %s", md_files[0], exc)

    blocks: Optional[list[dict]] = None
    # 显式排除 *_content_list_v2.json：v2 是 list[list[dict]]（按页分组的嵌套结构），
    # 误选会让 blocks 非空但形状错误（downstream blocks_to_sections 拿到内层 list 当 dict 用 →
    # sections 全空）。只要 v1（content_list.json，list[dict]）存在就优先它。
    json_files = sorted(
        (p for p in workdir.glob("*content_list*.json") if "_v2" not in p.stem),
        key=lambda p: (p.name != "content_list.json", p.name),
    )
    if json_files:
        try:
            loaded = json.loads(json_files[0].read_text(encoding="utf-8"))
            # 形状守卫（engine-agnostic）：blocks 契约是 list[dict]；v2 是 list[list[dict]]，
            # 首元素是 list 而非 dict → 拒绝（即使文件名过滤漏网也不误读成错形状）。
            if isinstance(loaded, list) and (not loaded or isinstance(loaded[0], dict)):
                blocks = loaded
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取 content_list 失败 %s: %s", json_files[0], exc)

    images_dir = workdir / "images"
    # 绝对路径：markdown 里的图片引用是相对 images/xxx，消费者用 asset_dir / 相对名拼接定位文件
    asset_dir = images_dir.resolve() if images_dir.is_dir() else None

    return markdown, blocks, asset_dir


__all__ = [
    "MANIFEST_FILENAME",
    "source_hash_from_path",
    "signature_dir",
    "is_ready",
    "lookup",
    "reserve",
    "write_manifest",
    "cleanup_failed",
    "load_ir",
]
