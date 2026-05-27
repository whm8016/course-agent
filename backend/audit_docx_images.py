"""
按 image_extractor 现有逻辑统计 DOCX/PDF 能提取多少张图（不调 API、不写 LightRAG）。

在 backend 目录运行:

  python audit_docx_images.py
  python audit_docx_images.py "kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx"
  python audit_docx_images.py --dir kb_store/circuswithpic/raw
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.rag.llamaindex.image_extractor import (
    _MIN_IMAGE_AREA,
    _MIN_IMAGE_PX,
    _MAX_IMAGES_PER_FILE,
    _image_meets_threshold,
    _write_vision_ready_png,
    collect_image_candidates,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audit_docx_media(docx_path: Path, out_dir: Path) -> dict:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(docx_path))
    seen: set[str] = set()

    stats = {
        "total_rels": 0,
        "media_rels": 0,
        "by_ext": Counter(),
        "kept": [],
        "skipped": [],
    }

    for rel in doc.part.rels.values():
        stats["total_rels"] += 1
        target = getattr(rel, "target_ref", "") or ""
        if not target.startswith("media/"):
            continue

        stats["media_rels"] += 1
        ext = Path(target).suffix.lower() or "(none)"
        stats["by_ext"][ext] += 1

        try:
            blob = rel.target_part.blob
        except Exception as exc:
            stats["skipped"].append(
                {"name": target, "ext": ext, "reason": f"读取失败: {exc}"}
            )
            continue

        if not blob or len(blob) < 100:
            stats["skipped"].append(
                {"name": target, "ext": ext, "size": len(blob or b""), "reason": "blob 过小 (<100B)"}
            )
            continue

        blob_hash = _sha256_bytes(blob)
        if blob_hash in seen:
            stats["skipped"].append(
                {"name": target, "ext": ext, "reason": "重复 hash"}
            )
            continue

        suffix = ext if ext != "(none)" else ".png"
        out_name = out_dir / f"{docx_path.stem}_{Path(target).name}"
        saved = _write_vision_ready_png(blob, out_name)

        if saved is None:
            w, h = 0, 0
            try:
                from PIL import Image

                with Image.open(io.BytesIO(blob)) as im:
                    w, h = im.width, im.height
            except Exception:
                pass

            if w and h:
                if w < _MIN_IMAGE_PX or h < _MIN_IMAGE_PX:
                    reason = f"边长不足 (需>={_MIN_IMAGE_PX}px, 实际 {w}x{h})"
                elif w * h < _MIN_IMAGE_AREA:
                    reason = f"面积不足 (需>={_MIN_IMAGE_AREA}, 实际 {w * h})"
                else:
                    reason = "转换 PNG 失败 (WMF/EMF 等无法渲染)"
            else:
                reason = "无法打开/转换"

            stats["skipped"].append(
                {
                    "name": target,
                    "ext": ext,
                    "size": len(blob),
                    "wh": f"{w}x{h}" if w else "?",
                    "reason": reason,
                }
            )
            continue

        seen.add(blob_hash)
        try:
            from PIL import Image

            with Image.open(saved) as im:
                w, h = im.width, im.height
        except Exception:
            w, h = 0, 0

        stats["kept"].append(
            {
                "name": target,
                "ext": ext,
                "saved": str(saved),
                "wh": f"{w}x{h}",
                "area": w * h,
            }
        )

    return stats


def _print_docx_detail(docx_path: Path, out_dir: Path) -> int:
    print(f"\n{'=' * 72}")
    print(f"文件: {docx_path.name}")
    print(f"{'=' * 72}")

    stats = _audit_docx_media(docx_path, out_dir)

    print(f"doc.part.rels 总数: {stats['total_rels']}")
    print(f"其中 media/: {stats['media_rels']}")
    print("\n按扩展名:")
    for ext, n in stats["by_ext"].most_common():
        print(f"  {ext}: {n}")

    kept_by_ext = Counter(item["ext"] for item in stats["kept"])
    skip_by_reason = Counter(item["reason"].split(" (")[0] for item in stats["skipped"])

    print(f"\n保留: {len(stats['kept'])} 张")
    for ext, n in kept_by_ext.most_common():
        print(f"  {ext}: {n}")

    print(f"\n跳过: {len(stats['skipped'])} 个")
    for reason, n in skip_by_reason.most_common():
        print(f"  {reason}: {n}")

    print("\n--- 保留列表 ---")
    for i, item in enumerate(stats["kept"], 1):
        print(f"  [{i:02d}] {Path(item['name']).name:20s} {item['wh']:>12s}  area={item['area']}")

    print("\n--- 跳过样本 (前 15) ---")
    for item in stats["skipped"][:15]:
        wh = item.get("wh", "")
        print(f"  {Path(item['name']).name:20s} {wh:>8s}  {item['reason']}")

    if len(stats["skipped"]) > 15:
        print(f"  ... 还有 {len(stats['skipped']) - 15} 个未列出")

    return len(stats["kept"])


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 DOCX 图片提取（现有逻辑）")
    parser.add_argument(
        "file",
        nargs="?",
        default="kb_store/circuswithpic/raw/28042090_3. 电路分析基础实验教案.docx",
        help="DOCX 文件路径",
    )
    parser.add_argument("--dir", help="扫描目录下所有 docx/pdf")
    args = parser.parse_args()

    backend = Path(__file__).parent
    out_dir = backend / "lightrag_store" / "_audit_extract" / "extracted_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("当前过滤条件:")
    print(f"  IMAGE_INGEST_MIN_PX   = {_MIN_IMAGE_PX}")
    print(f"  IMAGE_INGEST_MIN_AREA = {_MIN_IMAGE_AREA}")
    print(f"  IMAGE_INGEST_MAX_PER_FILE = {_MAX_IMAGES_PER_FILE}")
    print(f"  规则: 宽>=MIN_PX 且 高>=MIN_PX 且 宽*高>=MIN_AREA")

    if args.dir:
        scan_dir = Path(args.dir)
        if not scan_dir.is_absolute():
            scan_dir = backend / scan_dir
        files = sorted(
            p for p in scan_dir.rglob("*")
            if p.suffix.lower() in {".docx", ".pdf"}
        )
    else:
        target = Path(args.file)
        if not target.is_absolute():
            target = backend / target
        if not target.is_file():
            sys.exit(f"文件不存在: {target}")
        files = [target]

    total_kept = 0
    docx_files = [f for f in files if f.suffix.lower() == ".docx"]
    pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]

    for docx_path in docx_files:
        total_kept += _print_docx_detail(docx_path.resolve(), out_dir)

    if pdf_files:
        print(f"\n{'=' * 72}")
        print("PDF 文件（走 collect_image_candidates 汇总）")
        print(f"{'=' * 72}")
        candidates = collect_image_candidates([str(p) for p in pdf_files], out_dir)
        print(f"PDF 保留: {len(candidates)} 张")
        total_kept += len(candidates)

    # 最终与 collect_image_candidates 对齐验证
    all_files = [str(f.resolve()) for f in files]
    candidates = collect_image_candidates(all_files, out_dir)
    per_file = Counter(Path(c.source_file).name for c in candidates)

    print(f"\n{'=' * 72}")
    print("汇总（collect_image_candidates 最终结果）")
    print(f"{'=' * 72}")
    print(f"源文件数: {len(files)}")
    print(f"最终保留: {len(candidates)} 张")
    for name, n in per_file.most_common():
        print(f"  {name}: {n}")
    print(f"\n提取出的 PNG 保存在: {out_dir}")


if __name__ == "__main__":
    main()
