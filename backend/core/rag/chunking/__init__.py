"""RAG 切块策略包（可插拔扩展点）。

每种切块策略一个模块，由 ingestion.parse_files 按 settings.chunking.strategy 分发：
  - sentence_splitter（默认，ingestion.py 内联）：LlamaIndex SentenceSplitter 按字数切。
  - ragflow_manual_docx：DOCX 走移植自 RAGFlow Manual 的结构化切块（标题层级栈 +
    表格原子化），非 DOCX 仍走 sentence_splitter。

未来 PDF 结构化解析可新增 pdf_docling 策略模块，分发结构不变。
"""
