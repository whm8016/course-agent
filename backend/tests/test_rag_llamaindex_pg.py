"""llamaindex_pg 后端单测：registry / indexer / retriever 的逻辑流程。

纯 mock 测试——不依赖真实 Postgres / embedding API / docling，验证：
- registry 能注册并取出 llamaindex_pg 后端
- indexer.index 复用 parse_files chunks → 写入 PGVectorStore，IndexResult 正确
- indexer.delete 按 course_id 清行（底层 SQL）
- retriever dense(DEFAULT)+sparse(SPARSE)→RRF 融合 → retrieve_context 拼接
- retriever 无命中返回空串

真实 PGVectorStore / embedding 连通性验证在 Docker（plan 指定的真环境，Windows venv
触发不了 docling）。本套测试全部 mock 重依赖，只验证编排逻辑与终态。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# asyncio_mode=auto（pyproject [tool.pytest]）自动处理 async 测试，无需 pytestmark


# ── 纯函数 ─────────────────────────────────────────────────────────────────────


class TestNodesToDocs:
    def test_basic_conversion(self):
        from core.rag.retriever.llamaindex_pg import _nodes_to_docs

        node = MagicMock()
        node.get_content.return_value = "hello world"
        node.node_id = "nid-1"
        node.score = 0.88
        node.metadata = {"file_path": "/data/a.pdf"}
        result = MagicMock(nodes=[node])

        docs = _nodes_to_docs(result)

        assert docs == [
            {
                "chunk_id": "nid-1",
                "content": "hello world",
                "score": 0.88,
                "file_path": "/data/a.pdf",
            }
        ]

    def test_empty_nodes(self):
        from core.rag.retriever.llamaindex_pg import _nodes_to_docs

        assert _nodes_to_docs(MagicMock(nodes=None)) == []
        assert _nodes_to_docs(MagicMock(nodes=[])) == []

    def test_skip_empty_content(self):
        from core.rag.retriever.llamaindex_pg import _nodes_to_docs

        empty_node = MagicMock()
        empty_node.get_content.return_value = ""
        empty_node.node_id = "x"
        full_node = MagicMock()
        full_node.get_content.return_value = "keep"
        full_node.node_id = "y"
        full_node.score = 0.1
        full_node.metadata = {}

        docs = _nodes_to_docs(MagicMock(nodes=[empty_node, full_node]))
        assert len(docs) == 1
        assert docs[0]["chunk_id"] == "y"


class TestFormatContexts:
    def test_truncation_at_max_chars(self):
        from core.rag.retriever.llamaindex_pg import _format_contexts

        ctx = [{"content": "x" * 100}]
        out = _format_contexts(ctx, limit=1, max_chars=30)
        assert len(out) <= 30

    def test_multiple_chunks_joined(self):
        from core.rag.retriever.llamaindex_pg import _format_contexts

        out = _format_contexts(
            [{"content": "a"}, {"content": "b"}], limit=5, max_chars=1000
        )
        assert out == "a\n\nb"

    def test_empty(self):
        from core.rag.retriever.llamaindex_pg import _format_contexts

        assert _format_contexts([], limit=5, max_chars=100) == ""
        assert _format_contexts([{"content": ""}], limit=5, max_chars=100) == ""


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_auto_register_llamaindex_pg(self):
        from core.rag import registry

        registry._auto_register("llamaindex_pg")
        assert "llamaindex_pg" in registry._retrievers
        assert "llamaindex_pg" in registry._indexers

    def test_get_indexer_returns_instance(self):
        from core.rag import get_indexer
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        indexer = get_indexer("llamaindex_pg")
        assert isinstance(indexer, LlamaIndexIndexer)

    def test_get_retriever_returns_instance(self):
        from core.rag import get_retriever
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = get_retriever("llamaindex_pg")
        assert isinstance(retriever, LlamaIndexRetriever)

    def test_is_backend_available_missing_dep(self):
        from core.rag.registry import is_backend_available

        with patch(
            "core.rag.llamaindex.pg_store.get_settings"
        ) as gs:
            gs.return_value.embedding.api_key.get_secret_value.return_value = ""
            ok, _ = is_backend_available("llamaindex_pg")
        assert ok is False


# ── indexer ───────────────────────────────────────────────────────────────────


class TestLlamaIndexIndexer:
    async def test_index_success(self):
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        indexer = LlamaIndexIndexer()
        with (
            patch("core.rag.ingestion.parse_files", return_value=(["c1", "c2"], ["s1", "s2"], {})),
            patch("core.rag.llamaindex.pg_store.get_vector_store", return_value=MagicMock()),
            patch("core.rag.llamaindex.pg_store.get_embed_model", return_value=MagicMock()),
            patch("llama_index.core.VectorStoreIndex") as vsi,
            patch("llama_index.core.StorageContext") as sc,
        ):
            sc.from_defaults.return_value = MagicMock()
            result = await indexer.index("course1", ["/data/a.pdf"])

        assert result.status == "success"
        assert result.chunks_created == 2
        assert result.files_indexed == 1
        assert result.course_id == "course1"
        vsi.assert_called_once()  # 确实触发了 embed+写入

    async def test_index_no_chunks_skipped(self):
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        indexer = LlamaIndexIndexer()
        with (
            patch("core.rag.ingestion.parse_files", return_value=([], [], {})),
            patch("core.rag.llamaindex.pg_store.get_vector_store"),
            patch("core.rag.llamaindex.pg_store.get_embed_model"),
        ):
            result = await indexer.index("course1", ["/data/a.pdf"])

        assert result.status == "skipped"
        assert result.chunks_created == 0

    async def test_index_resume_skips_first_n(self):
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        indexer = LlamaIndexIndexer()
        with (
            patch("core.rag.ingestion.parse_files", return_value=(["c1", "c2", "c3"], ["s1", "s2", "s3"], {})),
            patch("core.rag.llamaindex.pg_store.get_vector_store", return_value=MagicMock()),
            patch("core.rag.llamaindex.pg_store.get_embed_model", return_value=MagicMock()),
            patch("llama_index.core.VectorStoreIndex") as vsi,
            patch("llama_index.core.StorageContext") as sc,
        ):
            sc.from_defaults.return_value = MagicMock()
            # resume_from_chunk=1 → 跳过 c1，只写 c2/c3
            result = await indexer.index("course1", ["/a.pdf"], resume_from_chunk=1)

        assert result.status == "success"
        assert result.chunks_created == 2  # 跳过第1个
        # VectorStoreIndex 收到的 nodes 应为 2 个
        nodes_arg = vsi.call_args.kwargs.get("nodes")
        assert nodes_arg is not None and len(nodes_arg) == 2

    async def test_index_error_returns_error_status(self):
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        indexer = LlamaIndexIndexer()
        with patch("core.rag.ingestion.parse_files", side_effect=RuntimeError("boom")):
            result = await indexer.index("course1", ["/a.pdf"])

        assert result.status == "error"
        assert "boom" in (result.error or "")

    async def test_delete_runs_sql(self):
        from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer

        fake_conn = AsyncMock()
        fake_conn.execute = AsyncMock()

        @asynccontextmanager
        async def fake_begin():
            yield fake_conn

        fake_engine = MagicMock()
        fake_engine.begin = fake_begin

        with patch("core.db.database.engine", fake_engine):
            ok = await LlamaIndexIndexer().delete("course1")

        assert ok is True
        fake_conn.execute.assert_awaited_once()
        # SQL 应含 DELETE FROM data_kb_chunks + course_id 绑定
        sql_obj = fake_conn.execute.call_args.args[0]
        sql_text = str(sql_obj)
        assert "DELETE FROM data_kb_chunks" in sql_text
        # params 是 execute 的第二个位置参数（args[1]），非 kwargs
        assert fake_conn.execute.call_args.args[1] == {"cid": '{"course_id": "course1"}'}


# ── retriever ─────────────────────────────────────────────────────────────────


def _make_node(text="doc content", node_id="nid1", score=0.9):
    node = MagicMock()
    node.get_content.return_value = text
    node.node_id = node_id
    node.score = score
    node.metadata = {"file_path": "/data/a.pdf"}
    return node


class TestLlamaIndexRetriever:
    async def test_retrieve_context_fusion(self):
        """dense + sparse 各返回同一 node → RRF 融合 → 拼接进 context。"""
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = LlamaIndexRetriever()
        node = _make_node()

        fake_vs = MagicMock()

        async def fake_aquery(_q):
            r = MagicMock()
            r.nodes = [node]
            return r

        fake_vs.aquery = fake_aquery

        fake_emb = MagicMock()
        fake_emb._aget_query_embedding = AsyncMock(return_value=[0.1] * 8)

        with (
            patch("core.rag.llamaindex.pg_store.get_vector_store", return_value=fake_vs),
            patch("core.rag.llamaindex.pg_store.get_embed_model", return_value=fake_emb),
            patch(
                "core.rag.llamaindex.pg_store.is_llamaindex_pg_available",
                return_value=(True, ""),
            ),
            patch("core.rag.llamaindex.pg_store.course_filter", return_value=MagicMock()),
        ):
            ctx = await retriever.retrieve_context("course1", "query", top_k=5)

        assert "doc content" in ctx

    async def test_retrieve_context_empty_when_no_hits(self):
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = LlamaIndexRetriever()
        fake_vs = MagicMock()

        async def fake_aquery(_q):
            r = MagicMock()
            r.nodes = []
            return r

        fake_vs.aquery = fake_aquery
        fake_emb = MagicMock()
        fake_emb._aget_query_embedding = AsyncMock(return_value=[0.1] * 8)

        with (
            patch("core.rag.llamaindex.pg_store.get_vector_store", return_value=fake_vs),
            patch("core.rag.llamaindex.pg_store.get_embed_model", return_value=fake_emb),
            patch(
                "core.rag.llamaindex.pg_store.is_llamaindex_pg_available",
                return_value=(True, ""),
            ),
            patch("core.rag.llamaindex.pg_store.course_filter", return_value=MagicMock()),
        ):
            ctx = await retriever.retrieve_context("course1", "query", top_k=5)

        assert ctx == ""

    async def test_retrieve_context_unavailable_returns_empty(self):
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever

        retriever = LlamaIndexRetriever()
        with patch(
            "core.rag.llamaindex.pg_store.is_llamaindex_pg_available",
            return_value=(False, "no key"),
        ):
            ctx = await retriever.retrieve_context("course1", "query")

        assert ctx == ""

    async def test_retrieve_returns_retrieval_results(self):
        from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever
        from core.rag.types import RetrievalResult

        retriever = LlamaIndexRetriever()
        node = _make_node(text="fact chunk", node_id="nid9", score=0.7)
        fake_vs = MagicMock()

        async def fake_aquery(_q):
            r = MagicMock()
            r.nodes = [node]
            return r

        fake_vs.aquery = fake_aquery
        fake_emb = MagicMock()
        fake_emb._aget_query_embedding = AsyncMock(return_value=[0.1] * 8)

        with (
            patch("core.rag.llamaindex.pg_store.get_vector_store", return_value=fake_vs),
            patch("core.rag.llamaindex.pg_store.get_embed_model", return_value=fake_emb),
            patch(
                "core.rag.llamaindex.pg_store.is_llamaindex_pg_available",
                return_value=(True, ""),
            ),
            patch("core.rag.llamaindex.pg_store.course_filter", return_value=MagicMock()),
        ):
            results = await retriever.retrieve("course1", "query", top_k=3)

        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].content == "fact chunk"
        assert results[0].metadata["backend"] == "llamaindex_pg"

    async def test_sparse_store_degrades_on_exception(self):
        """_PgSparseStore.bm25_search 异常时返回 []（hybrid_retriever 据此跳过 sparse 路）。"""
        from core.rag.retriever.llamaindex_pg import _PgSparseStore

        fake_vs = MagicMock()

        async def boom(_q):
            raise RuntimeError("tsvector not ready")

        fake_vs.aquery = boom
        store = _PgSparseStore(fake_vs)
        out = await store.bm25_search("query", "course1", 10)
        assert out == []


# ── migration 016 ─────────────────────────────────────────────────────────────


class TestMigration016:
    def test_structure(self):
        """016 迁移结构校验（按文件路径加载，绕开 alembic.versions 包名与第三方 alembic 库同名）。"""
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parent.parent
            / "alembic"
            / "versions"
            / "016_kb_index_backend.py"
        )
        spec = importlib.util.spec_from_file_location("_m016", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # 顶层仅 from alembic import op + import sa，op 在函数内不触发
        assert mod.revision == "016"
        assert mod.down_revision == "015"
        assert callable(mod._table_exists)
        assert callable(mod._column_exists)
