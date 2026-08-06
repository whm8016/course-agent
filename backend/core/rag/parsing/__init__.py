"""解析层公共入口。

提供 ``parse_document``（查缓存→调引擎）+ ``ParsedDocument`` IR。所有引擎产出同一 IR，
消费者（``file_paths_to_llama_documents``）无需判断哪个引擎跑过。

默认引擎 ``mineru_api``（托管 API，云端不装 torch）。换引擎改 ``settings.parsing.engine``
一个配置项（registry 统一契约）。
"""
from core.rag.parsing.service import parse_document
from core.rag.parsing.types import ParsedDocument, ParserError

__all__ = ["parse_document", "ParsedDocument", "ParserError"]
