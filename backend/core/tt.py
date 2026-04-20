"""LightRAG 本地烟测。

- project（默认）：与线上相同 working_dir + workspace=course_{course_id}，适合已有 backend 建好的索引。
- upstream：与官方例程相同，仅单目录 working_dir（如 ./rag_storage），不区分 workspace。

已有索引时请勿再全量索引：默认不调用 index_course_with_lightrag；需要时设 LIGHTRAG_TT_INDEX=1。
"""
import asyncio
import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from lightrag.utils import setup_logger

from config import LIGHTRAG_QUERY_MODE
from core.lightrag_engine import (
    _build_query_param,
    _get_instance,
    index_course_with_lightrag,
    is_lightrag_available,
)

setup_logger("lightrag", level="INFO")

DEFAULT_COURSE_ID = os.getenv("LIGHTRAG_TT_COURSE_ID", "algorithm")
# project | upstream（官方例程：单目录 + openai_embed / gpt_4o_mini_complete）
TT_STYLE = os.getenv("LIGHTRAG_TT_STYLE", "project").strip().lower()


async def _run_project() -> None:
    rag = None
    ok, reason = is_lightrag_available()
    if not ok:
        print(f"LightRAG 不可用: {reason}")
        return

    course_id = (DEFAULT_COURSE_ID or "algorithm").strip()
    question = (os.getenv("LIGHTRAG_TT_QUESTION") or "解释一下背包问题").strip()
    mode = (os.getenv("LIGHTRAG_TT_MODE") or LIGHTRAG_QUERY_MODE or "hybrid").strip()

    insert_smoke = os.getenv("LIGHTRAG_TT_INSERT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    do_index = os.getenv("LIGHTRAG_TT_INDEX", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    try:
        rag = await _get_instance(course_id)
        if do_index:
            idx = await index_course_with_lightrag(course_id)
            print("index_course_with_lightrag:", idx)
        if insert_smoke:
            await rag.ainsert("解释一下背包问题")
        param = _build_query_param(mode, None, only_need_context=False)
        result = await rag.aquery(question, param=param)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag is not None:
            await rag.finalize_storages()


async def _run_upstream() -> None:
    """与 LightRAG 文档例程一致：单 working_dir，无 workspace。"""
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    rag = None
    raw_dir = os.getenv("LIGHTRAG_TT_RAG_STORAGE", "./rag_storage").strip()
    working_dir = (
        os.path.normpath(raw_dir)
        if os.path.isabs(raw_dir)
        else os.path.abspath(os.path.join(os.getcwd(), raw_dir))
    )
    os.makedirs(working_dir, exist_ok=True)

    question = (os.getenv("LIGHTRAG_TT_QUESTION") or "What are the top themes in this story?").strip()
    mode = (os.getenv("LIGHTRAG_TT_MODE") or "hybrid").strip()
    # 已有索引时设 LIGHTRAG_TT_INSERT=0；与文档完全一致要插入时再设 1
    insert_smoke = os.getenv("LIGHTRAG_TT_INSERT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    insert_text = os.getenv("LIGHTRAG_TT_INSERT_TEXT", "Your text").strip() or "Your text"

    try:
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        await rag.initialize_storages()
        if insert_smoke:
            await rag.ainsert(insert_text)
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag is not None:
            await rag.finalize_storages()


async def main() -> None:
    if TT_STYLE in ("upstream", "simple", "tutorial"):
        await _run_upstream()
    else:
        await _run_project()


if __name__ == "__main__":
    asyncio.run(main())
