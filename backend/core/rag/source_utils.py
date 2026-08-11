"""RAG 来源标识工具 —— chunk 文本突变的单一剥离入口（来源前缀 + chunk 后缀）。

两类突变都由摄入端（ingestion）注入、由下游剥离，构建/剥离成对维护于此，避免第二个
模块各自用正则镜像同一约定（日后调整规则只需改一处）：

- 来源前缀 ``【章节/来源/页码】\\n``：``ingestion._build_source_prefix`` 注入到 chunk 文本
  开头；``strip_source_prefix`` 剥掉，还原纯正文（供 contextual 定位等场景）。
- chunk 后缀 ``::chunk-<idx>``：为绕过 LightRAG filename 去重加到 file_path，检索端
  展示溯源前必须剥掉，否则用户会看到「xxx.pdf::chunk-5」（``strip_chunk_suffix``）。
"""
from __future__ import annotations

import re

_CHUNK_SUFFIX_MARKER = "::chunk-"
# 与 ingestion._build_source_prefix 成对：构建端拼 `【...】\n`，此处反操作剥掉。
_SOURCE_PREFIX_RE = re.compile(r"^【[^】]*】\n")


def strip_source_prefix(chunk: str) -> str:
    """剥离摄入时加的 `【章节/来源/页码】\n` 结构前缀，还原 chunk 纯正文。

    无前缀时原样返回。前缀格式与 ``ingestion._build_source_prefix`` 成对维护，单一真相源。
    """
    return _SOURCE_PREFIX_RE.sub("", chunk)


def strip_chunk_suffix(source: str) -> str:
    """剥离摄入时加的 `::chunk-<idx>` 后缀，还原真实来源文件路径。

    用 rfind 定位最后一个 `::chunk-` 后截断；无后缀（或出现在开头）时原样返回。
    """
    idx = source.rfind(_CHUNK_SUFFIX_MARKER)
    return source[:idx] if idx > 0 else source
