"""RAG/LightRAG 实例池与索引模块的修复回归测试。

覆盖：
- H-10：引用计数 evict 跳过 in_use 的并发 interleaving 推演（补充 test_concurrency_hardening
  的三个原语测试之外的真实调用链场景：lease_instance 上下文管理器配对释放、
  并发多 course 持有下 evict 全在用返回 None、超容不死循环）。
- M-22：finalize_storages 异常不再被 bare except 吞（evict 仍正常返回 id，错误进日志）。
- M-25：embedding 返回长度不匹配时抛错。
- M-26：is_lightrag_available 缺 embedding 配置时返回 False。
- M-27：file_routing 把 .doc/.ppt 归为 unsupported 并 warning；上传层 _ALLOWED_EXT 不含。
- M-28：resume 模式跳过图片重摄入。
- M-33：签名持久化 hydrate/persist/invalidate（Redis 可用 + 不可用降级）。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# H-10 补充：lease_instance 配对释放 + 并发超容不死循环
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h10_lease_instance_releases_on_success():
    """lease_instance 正常退出时 _release_instance 递减，实例可被 evict。"""
    from core.rag.lightrag.instance_pool import (
        lease_instance, _instances, _in_use, evict_oldest,
    )

    _instances.clear()
    _in_use.clear()

    mock_rag = SimpleNamespace(finalize_storages=AsyncMock())
    _instances["c1"] = mock_rag
    # 预置一个引用（模拟别处正在用），lease 再 +1 → 2
    _in_use["c1"] = 1

    async with lease_instance("c1") as rag:
        assert rag is mock_rag
        assert _in_use["c1"] == 2  # lease 进入时 +1

    # 离开 with：lease 释放一次 → 回到 1（仍有外部引用，不能 evict）
    assert _in_use["c1"] == 1
    result = await evict_oldest()
    assert result is None  # 仍在用，跳过
    assert "c1" in _instances


@pytest.mark.asyncio
async def test_h10_lease_instance_releases_on_exception():
    """lease_instance 异常退出时也保证 _release_instance（finally 语义）。"""
    from core.rag.lightrag.instance_pool import (
        lease_instance, _instances, _in_use,
    )

    _instances.clear()
    _in_use.clear()

    mock_rag = SimpleNamespace(finalize_storages=AsyncMock())
    _instances["c1"] = mock_rag

    with pytest.raises(ValueError, match="boom"):
        async with lease_instance("c1"):
            assert _in_use["c1"] == 1
            raise ValueError("boom")

    # 异常路径也释放了引用
    assert _in_use.get("c1") is None
    _instances.clear()


@pytest.mark.asyncio
async def test_h10_evict_skips_in_use_finds_next_free(monkeypatch):
    """OrderedDict 中第一个在用、第二个空闲 → evict 第二个（保 LRU 顺序找首个可用）。"""
    from core.rag.lightrag.instance_pool import (
        evict_oldest, _instances, _in_use,
    )

    _instances.clear()
    _in_use.clear()

    mock_old = SimpleNamespace(finalize_storages=AsyncMock())  # 最旧但在用
    mock_new = SimpleNamespace(finalize_storages=AsyncMock())  # 较新但空闲
    _instances["c_old"] = mock_old
    _instances["c_new"] = mock_new
    _in_use["c_old"] = 1  # 最旧在用

    result = await evict_oldest()
    assert result == "c_new"  # 跳过在用的，淘汰第二个
    assert "c_old" in _instances
    assert "c_new" not in _instances
    mock_old.finalize_storages.assert_not_called()
    mock_new.finalize_storages.assert_awaited_once()

    _instances.clear()
    _in_use.clear()


# ---------------------------------------------------------------------------
# M-22：finalize_storages 异常不再被静默吞
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m22_finalize_exception_does_not_crash_evict(caplog):
    """finalize_storages 抛 RuntimeError 时，evict 仍返回 id，异常进日志（不吞）。"""
    from core.rag.lightrag.instance_pool import (
        evict_oldest, _instances, _in_use,
    )

    _instances.clear()
    _in_use.clear()

    mock_rag = SimpleNamespace(
        finalize_storages=AsyncMock(side_effect=RuntimeError("storage down")),
    )
    _instances["c1"] = mock_rag

    with caplog.at_level("WARNING"):
        result = await evict_oldest()

    assert result == "c1"  # 仍正常淘汰
    assert "c1" not in _instances
    assert any("finalize_storages" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# M-25：embedding 长度校验
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m25_embedding_length_mismatch_raises():
    """provider 返回向量数少于 batch → 抛 RuntimeError，不写残缺结果。"""
    from core.rag.llamaindex.embedding_bridge import DashScopeEmbeddingClient

    client = DashScopeEmbeddingClient()

    # 构造一个返回 1 个向量（但 batch 有 2 条）的假 resp
    fake_item = SimpleNamespace(embedding=[0.1, 0.2], index=0)
    fake_resp = SimpleNamespace(data=[fake_item])  # 只有 1 个，batch=2

    fake_create = AsyncMock(return_value=fake_resp)
    fake_embeddings = SimpleNamespace(create=fake_create)
    fake_openai = SimpleNamespace(embeddings=fake_embeddings)

    with patch("core.rag.llamaindex.embedding_bridge._async_openai_client", fake_openai):
        with pytest.raises(RuntimeError, match="返回长度不匹配"):
            await client.embed(["text1", "text2"])


@pytest.mark.asyncio
async def test_m25_embedding_length_ok():
    """正常等长返回时不抛错（回归保护）。"""
    from core.rag.llamaindex.embedding_bridge import DashScopeEmbeddingClient

    client = DashScopeEmbeddingClient()
    fake_resp = SimpleNamespace(data=[
        SimpleNamespace(embedding=[0.1, 0.2], index=0),
        SimpleNamespace(embedding=[0.3, 0.4], index=1),
    ])
    fake_create = AsyncMock(return_value=fake_resp)
    fake_embeddings = SimpleNamespace(create=fake_create)
    fake_openai = SimpleNamespace(embeddings=fake_embeddings)

    with patch("core.rag.llamaindex.embedding_bridge._async_openai_client", fake_openai):
        vecs = await client.embed(["a", "b"])
    assert len(vecs) == 2


# ---------------------------------------------------------------------------
# M-26：可用性检查含 embedding 配置
# ---------------------------------------------------------------------------

def test_m26_available_returns_false_without_embedding(monkeypatch):
    """缺 EMBEDDING__MODEL 时 is_lightrag_available 返回 False（配置期 fail-fast）。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "LIGHTRAG_ENABLED", True)
    monkeypatch.setattr(llm_adapter, "LIGHTRAG_IMPORT_ERROR", None)
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_MODEL", "some-llm")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_API_KEY", "sk-xxx")
    monkeypatch.setattr(llm_adapter, "INDEX_EMBEDDING_MODEL", "")  # 缺 embedding model
    monkeypatch.setattr(llm_adapter, "INDEX_EMBEDDING_API_KEY", "sk-emb")

    ok, reason = llm_adapter.is_lightrag_available()
    assert ok is False
    assert "EMBEDDING__MODEL" in reason


def test_m26_available_returns_false_without_embedding_key(monkeypatch):
    """缺 EMBEDDING__API_KEY 时返回 False。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "LIGHTRAG_ENABLED", True)
    monkeypatch.setattr(llm_adapter, "LIGHTRAG_IMPORT_ERROR", None)
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_MODEL", "some-llm")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_API_KEY", "sk-xxx")
    monkeypatch.setattr(llm_adapter, "INDEX_EMBEDDING_MODEL", "text-embedding-v3")
    monkeypatch.setattr(llm_adapter, "INDEX_EMBEDDING_API_KEY", "")  # 缺 key

    ok, reason = llm_adapter.is_lightrag_available()
    assert ok is False
    assert "EMBEDDING__API_KEY" in reason


# ---------------------------------------------------------------------------
# M-27：.doc/.ppt 上传拒绝 + file_routing 归 unsupported
# ---------------------------------------------------------------------------

def test_m27_legacy_formats_classified_unsupported():
    """.doc/.ppt 不在 DOCX/PPTX 扩展集 → 归 unsupported（无 handler）。"""
    from core.rag.llamaindex.file_routing import FileTypeRouter, DocumentType

    assert FileTypeRouter.get_document_type("foo.doc") == DocumentType.UNKNOWN
    assert FileTypeRouter.get_document_type("foo.ppt") == DocumentType.UNKNOWN
    # 确认 OOXML 仍正常支持
    assert FileTypeRouter.get_document_type("foo.docx") == DocumentType.DOCX
    assert FileTypeRouter.get_document_type("foo.pptx") == DocumentType.PPTX


def test_m27_admin_allowed_ext_excludes_legacy():
    """上传层 _ALLOWED_EXT 不含 .doc/.ppt（根因修复）。"""
    from api.admin import _ALLOWED_EXT

    assert ".doc" not in _ALLOWED_EXT
    assert ".ppt" not in _ALLOWED_EXT
    assert ".docx" in _ALLOWED_EXT
    assert ".pptx" in _ALLOWED_EXT


# ---------------------------------------------------------------------------
# M-28：resume 跳过图片重摄入
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m28_resume_skips_image_ingest(monkeypatch):
    """resume_from_chunk>0 时，图片摄入分支被跳过（不调 ingest_images_from_files）。"""
    import core.rag.ingestion as ingestion_mod

    # _ingest_body 需要 rag 注入；mock 掉所有外部依赖
    called = {"image_ingest": False}

    async def _fake_ingest_images(*args, **kwargs):
        called["image_ingest"] = True
        return 0

    monkeypatch.setattr(
        "core.rag.lightrag.is_lightrag_available", lambda: (True, "")
    )

    # lease_instance 返回假 rag；不让它真去建 LightRAG
    class _FakeLease:
        async def __aenter__(self):
            return SimpleNamespace(workspace="ws")
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(
        "core.rag.lightrag.lease_instance", lambda cid: _FakeLease()
    )

    # parse_files 返回空 chunks（让函数走"解析结果为空"早返回，避开文本摄入）
    monkeypatch.setattr(ingestion_mod, "parse_files", lambda fps: ([], [], {}))
    monkeypatch.setattr(ingestion_mod, "_persist_lightrag_ingest_chunks", lambda *a, **k: None)

    # patch image_extractor 在模块内的延迟导入：让 ImportError 触发也会被吞，但我们要
    # 验证 resume 时连 import 都不发生——用 sys.modules 注入一个会记录调用的伪模块
    import sys
    fake_mod = SimpleNamespace(ingest_images_from_files=_fake_ingest_images)
    with patch.dict(sys.modules, {"core.rag.llamaindex.image_extractor": fake_mod}):
        await ingestion_mod.ingest_to_lightrag(
            course_id="c1", file_paths=["/tmp/x.pdf"],
            resume_from_chunk=5,  # resume 模式
        )

    assert called["image_ingest"] is False, "resume 模式不应触发图片摄入"


# ---------------------------------------------------------------------------
# M-33：签名持久化 hydrate/persist/invalidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m33_persist_then_hydrate_roundtrip(monkeypatch):
    """persist 写 Redis 后，清空内存再 hydrate 能读回原签名。"""
    from core.rag.lightrag import instance_pool

    store: dict[str, str] = {}
    fake_redis = MagicMock()
    fake_redis.set = AsyncMock(side_effect=lambda k, v, ex=None: store.__setitem__(k, v))
    fake_redis.get = AsyncMock(side_effect=lambda k: store.get(k))
    fake_redis.delete = AsyncMock(side_effect=lambda k: store.pop(k, None))

    monkeypatch.setattr(
        "core.db.cache._get_pool", lambda: fake_redis
    )

    sig = ("a|1|10", "b|2|20")
    instance_pool.set_cached_signature("c1", sig)
    await instance_pool.persist_signature("c1")

    # 清空内存（模拟重启）
    instance_pool._index_signatures.clear()
    assert instance_pool.get_cached_signature("c1") is None

    # hydrate 从 Redis 读回
    loaded = await instance_pool.hydrate_signature("c1")
    assert loaded == sig
    assert instance_pool.get_cached_signature("c1") == sig


@pytest.mark.asyncio
async def test_m33_hydrate_redis_unavailable_degrades(monkeypatch):
    """Redis 抛异常时 hydrate 不崩，返回 None（降级为纯内存）。"""
    from core.rag.lightrag import instance_pool

    def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr("core.db.cache._get_pool", _boom)

    instance_pool._index_signatures.clear()
    result = await instance_pool.hydrate_signature("c1")
    assert result is None  # 不抛，降级


@pytest.mark.asyncio
async def test_m33_invalidate_clears_both_layers(monkeypatch):
    """invalidate 同时清内存和 Redis。"""
    from core.rag.lightrag import instance_pool

    store: dict[str, str] = {"indexing:sig:c1": "[\"x\"]"}
    fake_redis = MagicMock()
    fake_redis.delete = AsyncMock(side_effect=lambda k: store.pop(k, None))
    monkeypatch.setattr("core.db.cache._get_pool", lambda: fake_redis)

    instance_pool.set_cached_signature("c1", ("x",))
    await instance_pool.invalidate_signature("c1")

    assert instance_pool.get_cached_signature("c1") is None
    assert "indexing:sig:c1" not in store
