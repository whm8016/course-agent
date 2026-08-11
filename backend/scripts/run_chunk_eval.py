"""One-off: parse_files + persist_ingest_chunks for chunk audit JSON."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# 必须在 import settings/ingestion 之前注入；空串 = 走 mupdf 本地 PDF 解析
if "--mupdf" in sys.argv:
    sys.argv.remove("--mupdf")
    os.environ["PARSING__ENGINE"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rag.ingestion import parse_files, persist_ingest_chunks
from settings import get_settings


def _node_id(course_id: str, text: str) -> str:
    return hashlib.md5(f"{course_id}|{text}".encode("utf-8")).hexdigest()


def run(course_id: str, file_paths: list[str]) -> Path | None:
    settings = get_settings()
    print(f"strategy={settings.chunking.strategy} ingest_size={settings.chunking.ingest_size}")
    print(f"parsing.engine={settings.parsing.engine!r} pdf.backend={settings.pdf.backend!r}")
    chunks, sources, doc_texts, _parse_errors = parse_files(file_paths)
    print(f"chunks={len(chunks)} doc_texts_keys={list(doc_texts.keys())}")
    node_ids = [_node_id(course_id, t) for t in chunks]
    out = persist_ingest_chunks(
        course_id,
        file_paths,
        chunks,
        0,
        sources,
        backend="llamaindex_pg",
        node_ids=node_ids,
    )
    print(f"saved: {out}")
    return out


if __name__ == "__main__":
    cid = sys.argv[1]
    fps = sys.argv[2:]
    run(cid, fps)
