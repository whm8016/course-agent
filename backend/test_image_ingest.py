"""
单张图片摄入测试（不跑全库几百张）

在 backend 目录下运行（需已配置 backend/.env 中的 DASHSCOPE_API_KEY）：

  # 0) 统计过滤后剩多少张（教案 raw media 常有 400+，过滤后约 60）
  python test_image_ingest.py audit --file "kb_store/circuswithpic/raw"

  # 1) 只看能从教案 docx 里抽出多少张图（不调 API、不写 LightRAG）
  python test_image_ingest.py collect --file "kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx"

  # 2) 只调 vision 描述第一张图（不写知识图谱，最快验效果）
  python test_image_ingest.py describe --limit 1 --file "kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx"

  # 3) 写入 LightRAG 知识图谱 1 张，并试检索
  python test_image_ingest.py ingest --course circuswithpic --limit 1 --file "kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx"
  python test_image_ingest.py ingest --course circuswithpic --limit 1 --query "实验电路图有哪些元件"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# describe/collect 单张测试时可设 IMAGE_INGEST_MAX_PER_FILE=1；audit 会读真实默认

from config import DASHSCOPE_API_KEY, LIGHTRAG_WORKDIR
from core.rag.llamaindex.image_extractor import (
    _RAGANYTHING_AVAILABLE,
    _make_vision_caption_func,
    collect_image_candidates,
    ingest_images_from_files,
)
from raganything.modalprocessors import ImageModalProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("test_image_ingest")

DEFAULT_DOCX = (
    Path(__file__).parent
    / "kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx"
)


def _resolve_file(path_str: str | None) -> Path:
    if not path_str:
        p = DEFAULT_DOCX
    else:
        p = Path(path_str)
        if not p.is_absolute():
            p = Path(__file__).parent / p
    if not p.is_file():
        raise SystemExit(f"文件不存在: {p}")
    return p.resolve()


def _work_dir(course_id: str) -> Path:
    return Path(LIGHTRAG_WORKDIR) / f"course_{course_id}" / "ingest_chunks"


def cmd_audit(path: Path) -> None:
    """统计目录/文件经尺寸过滤后的图片数（不调 vision）。"""
    from core.rag.llamaindex.image_extractor import (
        _MIN_IMAGE_AREA,
        _MIN_IMAGE_PX,
        _MAX_IMAGES_PER_FILE,
    )

    if path.is_dir():
        files = [str(p) for p in sorted(path.rglob("*")) if p.is_file()]
    else:
        files = [str(path)]
    work = _work_dir("_test_audit")
    candidates = collect_image_candidates(files, work)
    print(f"\n过滤条件: MIN_PX={_MIN_IMAGE_PX} MIN_AREA={_MIN_IMAGE_AREA} MAX_PER_FILE={_MAX_IMAGES_PER_FILE}")
    print(f"源文件数: {len(files)} → 保留图片: {len(candidates)} 张\n")
    from collections import Counter

    per = Counter(Path(c.source_file).name for c in candidates)
    for name, n in per.most_common():
        print(f"  {name}: {n}")
    if not candidates:
        print("无图片通过过滤。可调低 IMAGE_INGEST_MIN_AREA 或检查是否仅有 .doc（需转 .docx）。")


def cmd_collect(file_path: Path, limit: int) -> None:
    work = _work_dir("_test_collect")
    candidates = collect_image_candidates([str(file_path)], work)
    print(f"\n源文件: {file_path.name}")
    print(f"抽出候选图: {len(candidates)} 张（IMAGE_INGEST_MAX_PER_FILE={os.getenv('IMAGE_INGEST_MAX_PER_FILE')}）\n")
    for i, c in enumerate(candidates[:limit]):
        print(f"[{i}] entity={c.entity_name}")
        print(f"    img_path={c.img_path}")
        print(f"    source={Path(c.source_file).name} page={c.page_no}")
    if not candidates:
        print("未抽到图片。可换 .docx 或检查文档是否真有嵌入图。")
        sys.exit(1)


async def cmd_describe(file_path: Path, limit: int, use_cache: bool) -> None:
    if not DASHSCOPE_API_KEY:
        raise SystemExit("缺少 DASHSCOPE_API_KEY，请在 backend/.env 配置")
    if not _RAGANYTHING_AVAILABLE:
        raise SystemExit("ImageModalProcessor 不可用，检查 core/rag/vendor/raganything")

    work = _work_dir("_test_describe")
    cache_path = work / "image_desc_cache_test.json" if use_cache else None
    candidates = collect_image_candidates([str(file_path)], work)
    if not candidates:
        raise SystemExit("未抽到图片")
    candidates = candidates[:limit]

    desc_cache: dict[str, str] = {}
    cache_lock = asyncio.Lock()
    caption_func = _make_vision_caption_func(desc_cache, cache_lock)
    # describe 不需要真实 LightRAG 实例，传 None 会报错；用 generate_description_only 需 processor
    # ImageModalProcessor 构造需要 lightrag — 用最小 mock 不行，必须 _get_instance
    from core.rag.lightrag_engine import _get_instance

    rag = await _get_instance("_test_describe_only")
    processor = ImageModalProcessor(lightrag=rag, modal_caption_func=caption_func)

    for i, c in enumerate(candidates):
        print(f"\n=== 描述第 {i + 1} 张: {c.entity_name} ===")
        modal = {
            "img_path": c.img_path,
            "image_caption": c.captions,
            "image_footnote": [],
        }
        desc, entity_info = await processor.generate_description_only(
            modal_content=modal,
            content_type="image",
            entity_name=c.entity_name,
        )
        print("\n--- detailed_description（前 800 字）---")
        print((desc or "")[:800])
        print("\n--- entity_info ---")
        print(json.dumps(entity_info, ensure_ascii=False, indent=2))
    if cache_path and desc_cache:
        from core.rag.llamaindex.image_extractor import _save_desc_cache
        _save_desc_cache(cache_path, desc_cache)
        print(f"\n缓存已写入: {cache_path}")


async def cmd_ingest(
    course_id: str,
    file_path: Path,
    limit: int,
    use_cache: bool,
    query: str | None,
) -> None:
    if not DASHSCOPE_API_KEY:
        raise SystemExit("缺少 DASHSCOPE_API_KEY")
    from core.rag.lightrag_engine import _get_instance, retrieve_with_lightrag

    rag = await _get_instance(course_id)
    cache_path = _work_dir(course_id) / "image_desc_cache.json"
    if not use_cache and cache_path.is_file():
        print(f"提示: 使用已有缓存 {cache_path}；若要强制重调 API 加 --no-cache")

    n = await ingest_images_from_files(
        [str(file_path)],
        rag,
        cache_path=str(cache_path) if use_cache else None,
        max_images=limit,
        semaphore_limit=1,
    )
    print(f"\n写入知识图谱: {n} 张图片")

    if query:
        print(f"\n=== 检索测试: {query!r} ===")
        ctx = await retrieve_with_lightrag(course_id, query, mode="hybrid")
        print(f"\n--- contexts ({len(ctx)} 条，不含 LLM 生成答案) ---")
        for j, c in enumerate(ctx[:8]):
            text = c if isinstance(c, str) else str(c)
            print(f"[{j}] {text[:500]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="单张图片 LightRAG 摄入测试")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="统计过滤后图片数（推荐先跑）")
    p_audit.add_argument("--file", default="kb_store/circuswithpic/raw", help="文件或目录")

    p_collect = sub.add_parser("collect", help="只统计能抽出多少张图")
    p_collect.add_argument("--file", default=None, help="PDF/DOCX 路径")
    p_collect.add_argument("--limit", type=int, default=5, help="打印前 N 条候选")

    p_desc = sub.add_parser("describe", help="vision 描述，不写图谱")
    p_desc.add_argument("--file", default=None)
    p_desc.add_argument("--limit", type=int, default=1)
    p_desc.add_argument("--no-cache", action="store_true")

    p_ing = sub.add_parser("ingest", help="写入 1 张到 LightRAG")
    p_ing.add_argument("--course", default="circuswithpic")
    p_ing.add_argument("--file", default=None)
    p_ing.add_argument("--limit", type=int, default=1)
    p_ing.add_argument("--no-cache", action="store_true")
    p_ing.add_argument("--query", default=None, help="写入后试一条检索")

    args = parser.parse_args()
    if args.cmd == "audit":
        audit_path = Path(args.file)
        if not audit_path.is_absolute():
            audit_path = Path(__file__).parent / audit_path
        if not audit_path.exists():
            raise SystemExit(f"路径不存在: {audit_path}")
        cmd_audit(audit_path.resolve())
        return

    file_path = _resolve_file(getattr(args, "file", None))

    if args.cmd == "collect":
        cmd_collect(file_path, args.limit)
    elif args.cmd == "describe":
        asyncio.run(cmd_describe(file_path, args.limit, use_cache=not args.no_cache))
    elif args.cmd == "ingest":
        asyncio.run(
            cmd_ingest(
                args.course,
                file_path,
                args.limit,
                use_cache=not args.no_cache,
                query=args.query,
            )
        )


if __name__ == "__main__":
    main()
