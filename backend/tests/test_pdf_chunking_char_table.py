"""PDF 切块：字符计数 + 表格原子化回归测试。

回归两个 bug（详见 plan「修复 PDF 切块 token/字符 bug」）：
1. SentenceSplitter 的 chunk_size 默认按 tiktoken **token** 计数（_token_size=len(tokenizer(text))），
   导致 chunk 实际字符数远超 ``INGEST_SIZE``（英文 ~4 倍）。传 ``tokenizer=lambda t: t`` 后
   应按**字符**计数——本文件用 monkeypatch 把 INGEST_CHUNK_SIZE 调到极小值强制切多块验证。
2. PDF/MinerU markdown 内嵌 HTML ``<table>`` 被 SentenceSplitter 当纯文本，从单元格内部
   锯断数值（实测 AutoAct 论文 ``49.09`` → ``49.``/``09``）。表格应原子化，数值不得跨 chunk 断开。
"""
from __future__ import annotations

from llama_index.core.schema import Document

from core.rag.ingestion import (
    _atomize_table,
    _chunk_by_sentence_splitter,
    _split_markdown_by_tables,
)


def _long_text(n_chars: int) -> str:
    """生成约 n_chars 字符的英文句子流（带句号，便于 SentenceSplitter 切句）。"""
    unit = "The quick brown fox jumps over the lazy dog. "
    reps = n_chars // len(unit) + 1
    return (unit * reps)[:n_chars]


# ── 问题 1：字符计数 ──────────────────────────────────────────────────────────


def test_sentence_splitter_counts_chars_not_tokens(monkeypatch):
    """chunk_size=200（字符）下，每块内容 ≤200 字符（token 模式下英文会到 ~800）。"""
    import core.rag.ingestion as ing

    monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 200)
    monkeypatch.setattr(ing, "INGEST_CHUNK_OVERLAP", 20)
    doc = Document(text=_long_text(2000), metadata={})
    chunks, sources = _chunk_by_sentence_splitter([doc])

    assert len(chunks) > 1, "2000 字符文本在 200 字符阈值下应切成多块"
    # 字符计数模式：每块严格 ≤ chunk_size 字符；token 模式则普遍 4 倍超标
    over = [len(c) for c in chunks if len(c) > 200]
    assert not over, f"字符计数模式下每块应 ≤200 字符，超标块长度: {over}"
    # source 全局编号连续且配对
    assert len(chunks) == len(sources)
    assert all(s.endswith(f"::chunk-{i}") for i, s in enumerate(sources))


def test_section_prefix_preserved_after_restructure(monkeypatch):
    """重构后仍正确注入【章节: ...】前缀（回归：metadata 处理未被表格拆段破坏）。"""
    import core.rag.ingestion as ing

    monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 1000)
    monkeypatch.setattr(ing, "INGEST_CHUNK_OVERLAP", 50)
    doc = Document(
        text=_long_text(300),
        metadata={"file_path": "/tmp/p.pdf", "section": "实验三", "file_name": "p.pdf"},
    )
    chunks, _ = _chunk_by_sentence_splitter([doc])
    assert chunks, "应至少产出一个 chunk"
    # 正文段走 SentenceSplitter，每个 chunk 都带章节前缀
    assert all(c.startswith("【章节: 实验三】") for c in chunks), (
        f"重构后章节前缀丢失: {[c[:20] for c in chunks]}"
    )


# ── 问题 2：表格原子化 ────────────────────────────────────────────────────────


def test_split_markdown_by_tables_order():
    """正文/表格/正文交错分段，顺序与 is_table 标记正确。"""
    md = "intro before table.\n<table><tr><td>A</td></tr></table>\nafter table."
    segs = _split_markdown_by_tables(md)
    assert segs == [
        ("intro before table.\n", False),
        ("<table><tr><td>A</td></tr></table>", True),
        ("\nafter table.", False),
    ]


def test_split_markdown_by_tables_no_table_passthrough():
    """无表格的纯文本整段返回（is_table=False），等价原直通。"""
    assert _split_markdown_by_tables("just plain text, no tables here.") == [
        ("just plain text, no tables here.", False)
    ]


def test_atomize_table_small_single_chunk():
    """小表格整表一块，不切。"""
    t = "<table><tr><td>H</td></tr><tr><td>1</td></tr></table>"
    assert _atomize_table(t, 900) == [t]


def test_atomize_table_single_row_oversize_kept_whole():
    """单行就超阈值：无法再切，整表保留（宁超不断）。"""
    big = "<table><tr><td>" + "x" * 200 + "</td></tr></table>"
    assert _atomize_table(big, 80) == [big]


def test_atomize_table_splits_by_rows_with_header():
    """超长表按 <tr> 分组，每组带首行表头，行标签成对（不在行中间断）。"""
    header = "<tr><td>Metric</td><td>Value</td></tr>"
    rows = "".join(f"<tr><td>m{i}</td><td>{i}</td></tr>" for i in range(20))
    t = f"<table>{header}{rows}</table>"
    groups = _atomize_table(t, max_chars=120)

    assert len(groups) > 1, "超长表应按行分组切成多块"
    for g in groups:
        assert g.startswith("<table>") and g.endswith("</table>")
        assert header in g, "每个分组应重复首行表头"
        # 不在 <tr> 中间断：开闭标签必须成对
        assert g.count("<tr>") == g.count("</tr>"), "行标签必须成对（未在行中间断）"


def test_table_value_not_split_across_chunks(monkeypatch):
    """回归 AutoAct：表格数值 49.0x 不得被切成 49.0 / x 跨 chunk。"""
    import core.rag.ingestion as ing

    monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 80)  # 极小阈值强制表格按行分组
    monkeypatch.setattr(ing, "INGEST_CHUNK_OVERLAP", 0)
    header = "<tr><td>Method</td><td>Score</td></tr>"
    body = "".join(f"<tr><td>m{i}</td><td>49.0{i}</td></tr>" for i in range(1, 9))
    table = f"<table>{header}{body}</table>"
    doc = Document(text=f"see results below.\n{table}\nend.", metadata={})
    chunks, _ = _chunk_by_sentence_splitter([doc])

    # 每个数值必须完整出现在某个 chunk；旧逻辑会从单元格内部锯断 → 整数不可寻
    for i in range(1, 9):
        val = f"49.0{i}"
        assert any(val in c for c in chunks), f"数值 {val} 被切断或丢失（应完整出现在某 chunk）"


def test_inline_table_kept_atomic_with_surrounding_text(monkeypatch):
    """正文夹表格：表格整块、前后正文各自成块，顺序保留。"""
    import core.rag.ingestion as ing

    monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 1000)
    monkeypatch.setattr(ing, "INGEST_CHUNK_OVERLAP", 0)
    table = "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
    doc = Document(text=f"leading text.\n{table}\ntrailing text.", metadata={})
    chunks, _ = _chunk_by_sentence_splitter([doc])

    # 表格整体落在单个 chunk（A、B 同块）
    table_chunks = [c for c in chunks if "<table>" in c]
    assert len(table_chunks) == 1, "表格应原子化为单个 chunk"
    assert "<td>A</td>" in table_chunks[0] and "<td>B</td>" in table_chunks[0]
    # 前后正文都在
    assert any("leading text" in c for c in chunks)
    assert any("trailing text" in c for c in chunks)
