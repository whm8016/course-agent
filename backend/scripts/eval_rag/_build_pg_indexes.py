"""一次性摄入：把 mycourse 两份 DOCX 按两种切块策略各建一个 pgvector 索引。

对照设计（唯一变量 = strategy）：
  mycourse_pg_ss  sentence_splitter     固定切块（按句子+固定 size）
  mycourse_pg_rf  ragflow_manual_docx   结构化切块（标题层级栈+表格原子化）
两路同 ingest_size=900、关 contextual_enrichment、开图片描述（仿真真实使用场景）。
第二路复用第一路的图片描述缓存，同图不重花 VLM。

跑法（依赖 .venv 在主仓库，代码在 worktree）：
  "D:/agent/course-agent/backend/.venv/Scripts/python.exe" \
      backend/scripts/eval_rag/_build_pg_indexes.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

# backend/ 成根（让 from core.rag / from settings 可 import）。
# _build_pg_indexes.py 在 backend/scripts/eval_rag/ 下，parents[2] = backend/。
BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv

load_dotenv(BACKEND / ".env")

from core.rag import get_indexer  # noqa: E402
from settings import get_settings  # noqa: E402

# 源文档在主仓库（worktree 无 kb_store）
RAW_DIR = Path(r"D:/agent/course-agent/backend/kb_store/mycourse/raw")
FILE_PATHS = [str(p) for p in sorted(RAW_DIR.glob("*.docx"))]

PAIRS = [
    ("mycourse_pg_ss", "sentence_splitter"),
    ("mycourse_pg_rf", "ragflow_manual_docx"),
]


def _img_cache_path(course_id: str) -> Path:
    s = get_settings()
    return Path(s.paths.ingest_chunks_dir) / f"course_{course_id}" / "image_desc_cache.json"


async def main() -> None:
    assert FILE_PATHS, f"未找到 mycourse 源文档: {RAW_DIR}"
    print(f"源文档({len(FILE_PATHS)}): {FILE_PATHS}")

    s = get_settings()
    s.chunking.contextual_enrichment = False      # 真实场景默认关
    s.chunking.inline_image_descriptions = True   # 仿真真实场景（.env 默认即开）
    s.chunking.ingest_size = 900                  # 控制变量，两路同 size
    indexer = get_indexer("llamaindex_pg")

    for i, (course_id, strategy) in enumerate(PAIRS):
        s.chunking.strategy = strategy            # get_settings() 单例即时切，仅本进程
        # 第二路复用第一路的图片描述缓存（file_paths 相同 → 同图 base64sha 相同，命中即跳 VLM）
        if i > 0:
            prev = _img_cache_path(PAIRS[0][0])
            cur = _img_cache_path(course_id)
            if prev.exists():
                cur.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prev, cur)
                print(f"复用图片描述缓存 -> course_{course_id}")
        await indexer.delete(course_id)           # 幂等清旧，防累加（PGVectorStore.add 不 upsert）
        result = await indexer.index(course_id, FILE_PATHS)
        print(
            f"{course_id} strategy={strategy}: "
            f"status={result.status} chunks={result.chunks_created} err={result.error}"
        )


if __name__ == "__main__":
    asyncio.run(main())
