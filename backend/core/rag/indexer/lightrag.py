"""LightRAG Indexer 实现。

实现 Indexer ABC，提供基于 LightRAG 的文档索引能力。
从 lightrag_engine.py 迁移的索引逻辑。
"""
from __future__ import annotations

import logging
from pathlib import Path

from settings import get_settings
KNOWLEDGE_DIR = get_settings().paths.knowledge_dir

from core.rag.types import IndexResult
from core.rag.indexer.base import Indexer
from core.rag.lightrag import (
    _get_instance,
    _release_instance,
    is_lightrag_available,
    _build_signature,
    get_cached_signature,
    set_cached_signature,
    hydrate_signature,
    persist_signature,
    invalidate_signature,
)

logger = logging.getLogger(__name__)


def _resolve_source_dir(course_id: str, source_dir: str | None = None) -> Path:
    """解析源目录路径。

    source_dir 非空时必须解析到课程知识目录 ``KNOWLEDGE_DIR / course_id`` 之内（防
    服务端任意文件读：客户端传绝对路径或 ``../`` 逃逸会把任意目录文件灌进课程 KB，
    之后经 RAG 检索泄露）。传外部路径 -> ValueError。默认取课程目录。
    """
    base = (Path(KNOWLEDGE_DIR) / course_id).resolve()
    if source_dir:
        resolved = Path(source_dir).expanduser().resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"source_dir 必须在课程目录内: {base}")
        return resolved
    return base


def _collect_course_docs(source_dir: Path, course_id: str) -> tuple[list[str], list[str], list[str]]:
    """收集课程文档。

    Returns:
        (docs, ids, file_paths) 元组
    """
    if not source_dir.is_dir():
        return [], [], []

    docs: list[str] = []
    ids: list[str] = []
    file_paths: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue
        if not content.strip():
            continue
        docs.append(content)
        ids.append(f"{course_id}:{path.relative_to(source_dir).as_posix()}")
        file_paths.append(str(path.resolve()))
    return docs, ids, file_paths


class LightRAGIndexer(Indexer):
    """LightRAG 索引器实现。

    使用 LightRAG 知识图谱进行文档索引，支持增量更新。
    """

    async def index(
        self,
        course_id: str,
        file_paths: list[str],
        **kwargs,
    ) -> IndexResult:
        """索引文档文件。

        Args:
            course_id: 课程 ID
            file_paths: 待索引文件路径列表（可选，不传则扫描 KNOWLEDGE_DIR）
            **kwargs:
                - force: 是否强制重新索引
                - source_dir: 自定义源目录

        Returns:
            索引结果摘要
        """
        ok, reason = is_lightrag_available()
        if not ok:
            return IndexResult(
                course_id=course_id,
                files_indexed=0,
                chunks_created=0,
                status="error",
                error=reason,
            )

        force = kwargs.get("force", False)
        source_dir = kwargs.get("source_dir")

        try:
            rag = await _get_instance(course_id)
            resolved_dir = _resolve_source_dir(course_id, source_dir)

            if not resolved_dir.is_dir():
                return IndexResult(
                    course_id=course_id,
                    files_indexed=0,
                    chunks_created=0,
                    status="error",
                    error="source_dir_not_found",
                )

            # 确定要索引的文件
            if file_paths:
                all_files = file_paths
            else:
                all_files = [str(p.resolve()) for p in sorted(resolved_dir.rglob("*")) if p.is_file()]

            if not all_files:
                return IndexResult(
                    course_id=course_id,
                    files_indexed=0,
                    chunks_created=0,
                    status="skipped",
                    error="no_files",
                )

            # 检查签名缓存（M-33：先从 Redis hydrate 进内存，再读内存）
            signature = _build_signature(all_files)
            cache_key = f"{course_id}|{resolved_dir}"
            if not force:
                await hydrate_signature(cache_key)
            if not force and get_cached_signature(cache_key) == signature:
                logger.info("LightRAG index skipped (unchanged): course=%s", course_id)
                return IndexResult(
                    course_id=course_id,
                    files_indexed=0,
                    chunks_created=0,
                    status="skipped",
                    error="unchanged",
                )

            # 执行索引
            indexed_files = 0
            if hasattr(rag, "ainsert_files"):
                await rag.ainsert_files(all_files)
                indexed_files = len(all_files)

            docs, ids, text_file_paths = _collect_course_docs(resolved_dir, course_id)
            indexed_docs = 0
            if docs:
                await rag.ainsert(docs, ids=ids, file_paths=text_file_paths)
                indexed_docs = len(docs)

            # 更新签名缓存（M-33：内存 + Redis 持久化，重启后仍能命中跳过重索引）
            set_cached_signature(cache_key, signature)
            await persist_signature(cache_key)

            logger.info(
                "LightRAG indexed course=%s files=%d docs=%d source_dir=%s",
                course_id, indexed_files, indexed_docs, resolved_dir,
            )

            return IndexResult(
                course_id=course_id,
                files_indexed=indexed_files,
                chunks_created=indexed_docs,
                status="success",
            )

        except Exception as exc:
            logger.error("LightRAGIndexer.index failed: %s", exc, exc_info=True)
            return IndexResult(
                course_id=course_id,
                files_indexed=0,
                chunks_created=0,
                status="error",
                error=str(exc),
            )
        finally:
            # H-10：_get_instance 在本 try 内 +1，无论成功/异常都释放
            await _release_instance(course_id)

    async def delete(self, course_id: str) -> bool:
        """删除课程索引。

        注意：LightRAG 目前不支持删除单个 workspace，
        此方法仅清除签名缓存。
        """
        cache_key_prefix = f"{course_id}|"
        # 清除签名缓存（M-33：内存 + Redis 都清；原仅清内存，重启后 Redis 残留会误判"未变"）
        from core.rag.lightrag.instance_pool import _index_signatures
        keys_to_remove = [k for k in _index_signatures if k.startswith(cache_key_prefix)]
        for key in keys_to_remove:
            await invalidate_signature(key)

        logger.info("LightRAGIndexer.delete course=%s (signature cache cleared)", course_id)
        return True

    async def is_available(self) -> tuple[bool, str]:
        """检查 LightRAG 是否可用。"""
        return is_lightrag_available()


__all__ = [
    "LightRAGIndexer",
]
