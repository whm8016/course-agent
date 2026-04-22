from __future__ import annotations

import logging
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    RAG_BACKEND,
    TOP_K,
    VECTORSTORE_DIR,
)

logger = logging.getLogger(__name__)

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " "],
)

# ---------------------------------------------------------------------------
# Filesystem RAG (no Chroma / no native embedding runtime — stable on Windows)
# ---------------------------------------------------------------------------

_fs_cache: dict[str, list[tuple[str, str]]] = {}


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return {t for t in raw if len(t) > 1}


def _load_course_chunks_fs(course_id: str) -> list[tuple[str, str]]:
    if course_id in _fs_cache:
        return _fs_cache[course_id]

    course_dir = os.path.join(KNOWLEDGE_DIR, course_id)
    chunks: list[tuple[str, str]] = []
    if not os.path.isdir(course_dir):
        _fs_cache[course_id] = chunks
        return chunks

    for filename in sorted(os.listdir(course_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(course_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in _splitter.split_text(text):
            chunks.append((chunk, filename))

    _fs_cache[course_id] = chunks
    logger.info("FS RAG loaded %d chunks for course '%s'", len(chunks), course_id)
    return chunks


def _score_chunk(query_tokens: set[str], chunk: str) -> float:
    if not query_tokens:
        return 0.0
    ct = _tokenize(chunk)
    if not ct:
        return 0.0
    overlap = len(query_tokens & ct)
    return overlap / max(len(query_tokens), 1)


def retrieve_fs(course_id: str, query: str, top_k: int = TOP_K) -> list[dict]:
    chunks = _load_course_chunks_fs(course_id)
    if not chunks:
        return []

    qtok = _tokenize(query)
    scored: list[tuple[str, str, float]] = [
        (content, src, _score_chunk(qtok, content)) for content, src in chunks
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = [x for x in scored if x[2] > 0][:top_k]
    if not top:
        top = scored[:top_k]

    return [
        {"content": c, "source": s, "score": round(sc, 4)}
        for c, s, sc in top
    ]


def index_course_fs(course_id: str) -> int:
    _fs_cache.pop(course_id, None)
    return len(_load_course_chunks_fs(course_id))


# ---------------------------------------------------------------------------
# Chroma RAG (optional — lazy-import chromadb to avoid loading when using fs)
# ---------------------------------------------------------------------------

_chroma_client = None
_collections: dict = {}
_embedding_fn = None


def _chroma_ensure():
    global _chroma_client, _embedding_fn
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    if _embedding_fn is None:
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY required for Chroma RAG")
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=DASHSCOPE_API_KEY,
            api_base=DASHSCOPE_BASE_URL,
            model_name=EMBEDDING_MODEL,
        )


def _get_collection_chroma(course_id: str):
    _chroma_ensure()
    if course_id not in _collections:
        assert _chroma_client is not None and _embedding_fn is not None
        _collections[course_id] = _chroma_client.get_or_create_collection(
            name=f"course_{course_id}",
            embedding_function=_embedding_fn,
        )
    return _collections[course_id]


def index_course_chroma(course_id: str) -> int:
    course_dir = os.path.join(KNOWLEDGE_DIR, course_id)
    if not os.path.isdir(course_dir):
        return 0

    try:
        collection = _get_collection_chroma(course_id)
    except Exception:
        logger.exception("Chroma get_collection failed for course %s", course_id)
        return 0

    if collection.count() > 0:
        logger.info("Course '%s' already indexed (%d chunks)", course_id, collection.count())
        return collection.count()

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for filename in sorted(os.listdir(course_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(course_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        parts = _splitter.split_text(text)
        for i, chunk in enumerate(parts):
            chunk_id = f"{course_id}_{filename}_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append({"source": filename, "course": course_id})

    if not all_chunks:
        return 0

    logger.info("Indexing course '%s': %d chunks", course_id, len(all_chunks))

    try:
        batch_size = 20
        for start in range(0, len(all_chunks), batch_size):
            end = start + batch_size
            collection.add(
                documents=all_chunks[start:end],
                ids=all_ids[start:end],
                metadatas=all_meta[start:end],
            )
        logger.info("Indexed course '%s': %d chunks stored", course_id, len(all_chunks))
        return len(all_chunks)
    except Exception:
        logger.exception("Chroma index failed for course %s", course_id)
        return 0


def retrieve_chroma(course_id: str, query: str, top_k: int = TOP_K) -> list[dict]:
    try:
        collection = _get_collection_chroma(course_id)
        if collection.count() == 0:
            indexed_count = index_course_chroma(course_id)
            if indexed_count == 0 or collection.count() == 0:
                return []

        n = min(top_k, collection.count())
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - dist)
            chunks.append({
                "content": doc,
                "source": (meta or {}).get("source", ""),
                "score": round(score, 4),
            })
        logger.info("Retrieved %d chunks for course '%s'", len(chunks), course_id)
        return chunks
    except Exception:
        logger.exception("Chroma retrieve failed (course=%s)", course_id)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_course(course_id: str) -> int:
    if RAG_BACKEND == "chroma":
        return index_course_chroma(course_id)
    return index_course_fs(course_id)


def retrieve(course_id: str, query: str, top_k: int = TOP_K) -> list[dict]:
    if RAG_BACKEND == "chroma":
        out = retrieve_chroma(course_id, query, top_k)
        if not out and os.path.isdir(os.path.join(KNOWLEDGE_DIR, course_id)):
            logger.warning("Chroma returned empty; falling back to filesystem RAG")
            return retrieve_fs(course_id, query, top_k)
        return out
    return retrieve_fs(course_id, query, top_k)


def retrieve_texts(course_id: str, query: str, top_k: int = TOP_K) -> list[str]:
    chunks = retrieve(course_id, query, top_k)
    return [c["content"] for c in chunks]


def retrieve_context(course_id: str, query: str, top_k: int = TOP_K) -> dict[str, str]:
    """
    Merge top RAG chunks into one context string for agents.
    Return shape matches legacy rag_search: ``answer`` (concatenated text) and ``provider`` label.
    """
    chunks = retrieve(course_id, query, top_k=top_k)
    if not chunks:
        return {"answer": "", "provider": f"rag:{RAG_BACKEND}"}
    parts: list[str] = []
    for c in chunks:
        content = str(c.get("content", ""))
        src = str(c.get("source", "")).strip()
        parts.append(f"[{src}]\n{content}" if src else content)
    answer = "\n\n---\n\n".join(parts)
    return {"answer": answer, "provider": f"rag:{RAG_BACKEND}"}


def index_all_courses():
    if not os.path.isdir(KNOWLEDGE_DIR):
        return
    for course_id in os.listdir(KNOWLEDGE_DIR):
        course_path = os.path.join(KNOWLEDGE_DIR, course_id)
        if os.path.isdir(course_path):
            index_course(course_id)
