"""从 PDF/DOCX/独立图片提取嵌入图，经 RAG-Anything ImageModalProcessor 写入 LightRAG 知识图谱。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, VISION_MODEL
from core.rag.llamaindex.file_routing import FileTypeRouter

if TYPE_CHECKING:
    from lightrag import LightRAG

logger = logging.getLogger(__name__)

# Word 教案会把公式拆成大量 11×12 小图；默认过滤后约几十张「真图」/文件
_MIN_IMAGE_PX = int(os.getenv("IMAGE_INGEST_MIN_PX", "80"))
_MIN_IMAGE_AREA = int(os.getenv("IMAGE_INGEST_MIN_AREA", "15000"))
_MAX_IMAGES_PER_FILE = int(os.getenv("IMAGE_INGEST_MAX_PER_FILE", "120"))
_SEMAPHORE_LIMIT = int(os.getenv("IMAGE_INGEST_SEMAPHORE", "5"))
# WMF blob < 此字节数视为公式碎片，跳过（有意义的电路图通常 > 2000B）
_WMF_MIN_BLOB_SIZE = int(os.getenv("IMAGE_INGEST_WMF_MIN_BLOB", "2000"))


def _image_meets_threshold(width: int, height: int) -> bool:
    """宽高均 >= MIN_PX，且像素面积 >= MIN_AREA。"""
    if width < _MIN_IMAGE_PX or height < _MIN_IMAGE_PX:
        return False
    return width * height >= _MIN_IMAGE_AREA

try:
    from raganything.modalprocessors import ImageModalProcessor

    _RAGANYTHING_AVAILABLE = True
except ImportError:
    ImageModalProcessor = None  # type: ignore[assignment,misc]
    _RAGANYTHING_AVAILABLE = False


@dataclass(frozen=True)
class _ImageCandidate:
    img_path: str
    source_file: str
    page_no: int | None
    image_index: int
    entity_name: str
    captions: list[str]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mime_for_image_bytes(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _wmf_to_png_via_svg(blob: bytes, out_path: Path) -> Path | None:
    """WMF → SVG (wmf2svg) → PNG (rsvg-convert)。"""
    import re
    import shutil
    import subprocess

    wmf2svg_bin = shutil.which("wmf2svg")
    rsvg_bin = shutil.which("rsvg-convert")
    if not wmf2svg_bin or not rsvg_bin:
        return None

    dest = out_path.with_suffix(".png")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wmf", delete=False) as tmp_wmf:
            tmp_wmf.write(blob)
            tmp_wmf_path = tmp_wmf.name

        result = subprocess.run(
            [wmf2svg_bin, tmp_wmf_path],
            capture_output=True,
            timeout=15,
        )
        os.unlink(tmp_wmf_path)

        if result.returncode != 0 or not result.stdout:
            return None

        # wmf2svg 输出的中文文本是 GBK 编码，需要先 GBK→str 再传给 rsvg-convert
        raw = result.stdout
        try:
            svg_text = raw.decode("gbk")
        except (UnicodeDecodeError, LookupError):
            svg_text = raw.decode("utf-8", errors="replace")
        svg_clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", svg_text)
        svg_clean = svg_clean.replace("<polyline ", '<polyline fill="none" ')
        svg_clean = svg_clean.replace("<polygon ", '<polygon fill="none" ')

        rsvg_result = subprocess.run(
            [rsvg_bin, "--format", "png", "--background-color", "white", "-o", str(dest)],
            input=svg_clean.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        if rsvg_result.returncode != 0 or not dest.exists():
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest
    except Exception as exc:
        logger.debug("wmf2svg+rsvg-convert 失败 %s: %s", out_path.name, exc)
        return None


def _write_vision_ready_png(blob: bytes, out_path: Path) -> Path | None:
    """
    将 DOCX/PDF 嵌入图转为 DashScope 可识别的 PNG。
    Word 常把 WMF/EMF 存成 .png 扩展名，直接上传会报 illegal image format。
    """
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow 未安装，无法转换 WMF/EMF；请 pip install Pillow")
        if blob[:8] == b"\x89PNG\r\n\x1a\n" or blob[:3] == b"\xff\xd8\xff":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            return out_path
        return None

    try:
        with Image.open(BytesIO(blob)) as im:
            if not _image_meets_threshold(im.width, im.height):
                logger.debug(
                    "跳过过小图片 %dx%d（需边长>=%d 且面积>=%d）",
                    im.width,
                    im.height,
                    _MIN_IMAGE_PX,
                    _MIN_IMAGE_AREA,
                )
                return None
            rgb = im.convert("RGB")
            dest = out_path.with_suffix(".png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(dest, format="PNG")
            return dest
    except Exception as exc:
        logger.debug("Pillow 转 PNG 失败 %s: %s", out_path.name, exc)
        if blob[:8] == b"\x89PNG\r\n\x1a\n" or blob[:3] == b"\xff\xd8\xff":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            return out_path

        _WMF_MAGIC = (b"\xd7\xcd\xc6\x9a", b"\x01\x00\t\x00")
        _EMF_MAGIC = b"\x01\x00\x00\x00"
        _is_wmf = blob[:4] in _WMF_MAGIC or (blob[:2] == b"\x01\x00" and len(blob) > 3 and blob[2:4] == b"\x09\x00")
        _is_emf = blob[:4] == _EMF_MAGIC
        _suffix = out_path.suffix.lower()

        if not (_is_wmf or _is_emf or _suffix in (".wmf", ".emf")):
            return None

        # WMF blob 太小 = 公式碎片（单个字母/符号），直接跳过
        if len(blob) < _WMF_MIN_BLOB_SIZE:
            logger.debug("跳过过小 WMF blob (%dB < %dB): %s", len(blob), _WMF_MIN_BLOB_SIZE, out_path.name)
            return None

        # 路径 1: PyMuPDF 渲染
        try:
            import fitz as _fitz

            filetype = "wmf" if (_is_wmf or _suffix == ".wmf") else "emf"
            _doc = _fitz.open(stream=blob, filetype=filetype)
            if _doc.page_count > 0:
                _page = _doc[0]
                _mat = _fitz.Matrix(2.0, 2.0)
                _pix = _page.get_pixmap(matrix=_mat, alpha=False)
                w, h = _pix.width, _pix.height
                if not _image_meets_threshold(w, h):
                    logger.debug("跳过过小 WMF/EMF %dx%d: %s", w, h, out_path.name)
                    _doc.close()
                    return None
                dest = out_path.with_suffix(".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_pix.tobytes("png"))
                _doc.close()
                return dest
            _doc.close()
        except Exception as fitz_exc:
            logger.debug("fitz WMF/EMF 渲染失败 %s: %s", out_path.name, fitz_exc)

        # 路径 2: wmf2svg + cairosvg（Linux Docker 回退）
        if _is_wmf or _suffix == ".wmf":
            svg_result = _wmf_to_png_via_svg(blob, out_path)
            if svg_result is not None:
                return svg_result

        return None


def _load_desc_cache(cache_path: Path | None) -> dict[str, str]:
    if not cache_path or not cache_path.is_file():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取图片描述缓存 %s: %s", cache_path, exc)
    return {}


def _save_desc_cache(cache_path: Path | None, cache: dict[str, str]) -> None:
    if not cache_path:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("无法写入图片描述缓存 %s: %s", cache_path, exc)


def _make_vision_caption_func(
    desc_cache: dict[str, str],
    cache_lock: asyncio.Lock,
) -> Callable[..., Any]:
    """返回 ImageModalProcessor 所需的 async modal_caption_func（qwen-vl-plus）。"""

    async def vision_caption_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        image_data: str | None = None,
        messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        del history_messages  # RAG-Anything 接口占位

        cache_key: str | None = None
        if image_data:
            cache_key = _sha256_bytes(image_data.encode("utf-8") if isinstance(image_data, str) else image_data)
            async with cache_lock:
                if cache_key in desc_cache:
                    return desc_cache[cache_key]

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
        max_tokens = int(kwargs.pop("max_tokens", 2048))

        if messages:
            resp = await client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
        elif image_data:
            import base64

            raw = base64.b64decode(image_data)
            mime = _mime_for_image_bytes(raw)
            msgs: list[dict] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_data}"},
                    },
                ],
            })
            resp = await client.chat.completions.create(
                model=VISION_MODEL,
                messages=msgs,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            resp = await client.chat.completions.create(
                model=VISION_MODEL,
                messages=msgs,
                max_tokens=max_tokens,
                **kwargs,
            )

        content = resp.choices[0].message.content or ""
        if cache_key and content.strip():
            async with cache_lock:
                desc_cache[cache_key] = content
        return content

    return vision_caption_func


def _extract_pdf_images(
    pdf_path: str,
    out_dir: Path,
    seen_hashes: set[str] | None = None,
) -> list[_ImageCandidate]:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF 未安装，跳过 PDF 图片提取: %s", pdf_path)
        return []

    candidates: list[_ImageCandidate] = []
    seen = seen_hashes if seen_hashes is not None else set()
    src = Path(pdf_path).resolve()
    try:
        doc = fitz.open(str(src))
    except Exception as exc:
        logger.warning("无法打开 PDF %s: %s", src.name, exc)
        return []

    try:
        for page_no, page in enumerate(doc):
            if len(candidates) >= _MAX_IMAGES_PER_FILE:
                break
            seen_xrefs: set[int] = set()
            for img_index, img_info in enumerate(page.get_images()):
                if len(candidates) >= _MAX_IMAGES_PER_FILE:
                    break
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                except Exception as exc:
                    logger.debug("PDF 图片提取失败 %s p%d xref=%s: %s", src.name, page_no, xref, exc)
                    continue
                width = int(extracted.get("width") or 0)
                height = int(extracted.get("height") or 0)
                if not _image_meets_threshold(width, height):
                    continue
                ext = extracted.get("ext") or "png"
                image_bytes = extracted.get("image")
                if not image_bytes:
                    continue
                blob_hash = _sha256_bytes(image_bytes)
                if blob_hash in seen:
                    continue
                out_name = f"{src.stem}_p{page_no}_i{img_index}.{ext}"
                saved = _write_vision_ready_png(image_bytes, out_dir / out_name)
                if saved is not None:
                    seen.add(blob_hash)
                if saved is None:
                    continue
                out_path = saved
                entity_name = f"{src.stem}_第{page_no + 1}页_图{img_index + 1}"
                candidates.append(
                    _ImageCandidate(
                        img_path=str(out_path.resolve()),
                        source_file=str(src),
                        page_no=page_no,
                        image_index=img_index,
                        entity_name=entity_name,
                        captions=[f"第{page_no + 1}页插图"],
                    )
                )
    finally:
        doc.close()
    return candidates


def _extract_docx_images(
    docx_path: str,
    out_dir: Path,
    seen_hashes: set[str] | None = None,
) -> list[_ImageCandidate]:
    """从 DOCX 提取嵌入图。

    遍历 doc.part.rels 中所有 media/ 关系（覆盖表格、页眉等任意位置的图片），
    section 编号通过单独的段落扫描得到（best-effort）。
    """
    p = Path(docx_path).resolve()
    if p.suffix.lower() != ".docx":
        return []
    try:
        from docx import Document as DocxDocument
    except ImportError:
        logger.warning("python-docx 未安装，跳过 DOCX 图片提取: %s", p.name)
        return []

    candidates: list[_ImageCandidate] = []
    seen = seen_hashes if seen_hashes is not None else set()
    skipped_dup = 0
    skipped_small = 0

    try:
        doc = DocxDocument(str(p))
    except Exception as exc:
        logger.warning("无法打开 DOCX %s: %s", p.name, exc)
        return []

    _NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    _H1_SVALS = {"Heading1", "1", "2"}

    # 收集 Heading-1 出现位置（段落序号），用于给图片打节号
    heading_para_indices: list[int] = []
    for i, child in enumerate(doc.element.body):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag != "p":
            continue
        pstyle_el = child.find(f".//{{{_NS}}}pStyle")
        sval = (pstyle_el.get(f"{{{_NS}}}val") or "") if pstyle_el is not None else ""
        para_text = "".join(t.text or "" for t in child.iter(f"{{{_NS}}}t")).strip()
        if sval in _H1_SVALS and para_text:
            heading_para_indices.append(i)

    section_count = len(heading_para_indices)

    # 遍历所有 rels 中的 media/ 条目（覆盖段落、表格、页眉等所有位置）
    _ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".wmf", ".emf"}
    media_total = 0
    img_index = 0

    for rel in doc.part.rels.values():
        if len(candidates) >= _MAX_IMAGES_PER_FILE:
            break
        target = getattr(rel, "target_ref", "") or ""
        if not target.startswith("media/"):
            continue

        suffix = Path(target).suffix.lower() or ".png"
        if suffix not in _ALLOWED_SUFFIXES:
            continue

        media_total += 1
        try:
            blob = rel.target_part.blob
        except Exception as exc:
            logger.debug("DOCX 图片读取失败 %s: %s", p.name, exc)
            continue
        if not blob or len(blob) < 100:
            continue

        blob_hash = _sha256_bytes(blob)
        if blob_hash in seen:
            skipped_dup += 1
            continue

        saved = _write_vision_ready_png(blob, out_dir / f"{p.stem}_img{img_index}{suffix}")
        if saved is None:
            skipped_small += 1
            continue

        seen.add(blob_hash)
        section_idx = min(img_index // max(1, media_total // max(section_count, 1)), section_count)
        section_label = f"第{section_idx}节" if section_idx > 0 else ""
        candidates.append(
            _ImageCandidate(
                img_path=str(saved.resolve()),
                source_file=str(p),
                page_no=section_idx,
                image_index=img_index,
                entity_name=f"{p.stem}_{section_label}_嵌入图{img_index + 1}" if section_label else f"{p.stem}_嵌入图{img_index + 1}",
                captions=[f"{p.stem} {section_label} 嵌入图".strip()],
            )
        )
        img_index += 1

    logger.info(
        "DOCX 图片 %s: media=%d 保留=%d 跳过过小=%d 跳过重复=%d sections=%d",
        p.name,
        media_total,
        len(candidates),
        skipped_small,
        skipped_dup,
        section_count,
    )
    return candidates


def _candidate_pixel_area(img_path: str) -> int:
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return im.width * im.height
    except Exception:
        return 0


def _standalone_image_candidates(
    image_paths: list[str],
    seen_hashes: set[str] | None = None,
) -> list[_ImageCandidate]:
    candidates: list[_ImageCandidate] = []
    seen = seen_hashes if seen_hashes is not None else set()
    for idx, path_str in enumerate(image_paths):
        p = Path(path_str).resolve()
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
            h = _sha256_bytes(raw)
            if h in seen:
                continue
            saved = _write_vision_ready_png(raw, p)
            if saved is None:
                continue
            seen.add(h)
            use_path = str(saved.resolve())
        except OSError:
            continue
        candidates.append(
            _ImageCandidate(
                img_path=use_path,
                source_file=str(p),
                page_no=None,
                image_index=idx,
                entity_name=p.stem,
                captions=[p.name],
            )
        )
    return candidates


def collect_image_candidates(file_paths: list[str], work_dir: Path) -> list[_ImageCandidate]:
    """从文件列表收集待处理的图片候选（写入 work_dir 下的临时文件）。"""
    classification = FileTypeRouter.classify_files(file_paths)
    candidates: list[_ImageCandidate] = []
    seen_hashes: set[str] = set()

    extract_dir = work_dir / "extracted_images"
    for pdf_path in classification.parser_files:
        candidates.extend(_extract_pdf_images(pdf_path, extract_dir, seen_hashes))
    for docx_path in classification.docx_files:
        candidates.extend(_extract_docx_images(docx_path, extract_dir, seen_hashes))
    candidates.extend(_standalone_image_candidates(classification.image_files, seen_hashes))

    candidates.sort(key=lambda c: _candidate_pixel_area(c.img_path), reverse=True)

    logger.info(
        "图片候选收集完成: %d 张（来自 %d 个源文件；MIN_PX=%d MIN_AREA=%d）",
        len(candidates),
        len(file_paths),
        _MIN_IMAGE_PX,
        _MIN_IMAGE_AREA,
    )
    return candidates


def _lookup_doc_text(doc_texts: dict[str, list[str]] | None, source_file: str) -> list[str] | None:
    """按源文件路径查找文档分段文字列表（key 为 resolve 后的绝对路径）。"""
    if not doc_texts:
        return None
    resolved = str(Path(source_file).resolve())
    if resolved in doc_texts:
        return doc_texts[resolved]
    src = Path(source_file).resolve()
    for key, chunks in doc_texts.items():
        if Path(key).resolve() == src:
            return chunks
    return None


async def ingest_images_from_files(
    file_paths: list[str],
    rag: LightRAG,
    *,
    cache_path: str | None = None,
    doc_texts: dict[str, list[str]] | None = None,
    semaphore_limit: int = _SEMAPHORE_LIMIT,
    max_images: int | None = None,
    on_image_done: Callable[[int, int], Any] | None = None,
    control: Any | None = None,
) -> int:
    """
    提取嵌入图并通过 ImageModalProcessor 写入 LightRAG 知识图谱。

    Returns:
        成功处理的图片数量。
    """
    if not _RAGANYTHING_AVAILABLE or ImageModalProcessor is None:
        logger.warning("raganything 未安装，跳过图片知识图谱摄入")
        return 0

    if not file_paths:
        return 0

    cache_file = Path(cache_path) if cache_path else None
    desc_cache = _load_desc_cache(cache_file)
    cache_lock = asyncio.Lock()
    caption_func = _make_vision_caption_func(desc_cache, cache_lock)

    work_dir = cache_file.parent if cache_file else Path(tempfile.gettempdir()) / "course_agent_images"
    candidates = await asyncio.to_thread(collect_image_candidates, file_paths, work_dir)
    if not candidates:
        return 0
    if max_images is not None and max_images > 0:
        candidates = candidates[:max_images]
        logger.info("图片摄入上限 max_images=%d，仅处理前 %d 张", max_images, len(candidates))

    sem = asyncio.Semaphore(semaphore_limit)
    done_count = 0
    total = len(candidates)

    from core.rag.ingestion import IndexingAborted

    async def _process_one(candidate: _ImageCandidate, order: int) -> bool:
        nonlocal done_count
        modal_content = {
            "img_path": candidate.img_path,
            "image_caption": candidate.captions,
            "image_footnote": [],
        }
        async with sem:
            if control is not None:
                await control.checkpoint(chunks_done=order)
            try:
                proc = ImageModalProcessor(lightrag=rag, modal_caption_func=caption_func)
                context_chunks = _lookup_doc_text(doc_texts, candidate.source_file)
                if context_chunks:
                    proc.set_content_source(context_chunks, "text_chunks")
                chunk_index = candidate.page_no if candidate.page_no is not None else 0
                await proc.process_multimodal_content(
                    modal_content=modal_content,
                    content_type="image",
                    file_path=candidate.source_file,
                    entity_name=candidate.entity_name,
                    item_info={
                        "page_idx": chunk_index,
                        "index": chunk_index,
                    },
                    chunk_order_index=order,
                )
                done_count += 1
                if on_image_done:
                    result = on_image_done(done_count, total)
                    if asyncio.iscoroutine(result):
                        await result
                logger.info(
                    "图片已写入知识图谱: %s (%d/%d)",
                    candidate.entity_name,
                    done_count,
                    total,
                )
                return True
            except IndexingAborted:
                raise
            except Exception as exc:
                logger.warning(
                    "图片摄入失败（跳过） %s: %s",
                    candidate.entity_name,
                    exc,
                )
                return False

    tasks = [asyncio.create_task(_process_one(c, i)) for i, c in enumerate(candidates)]
    try:
        await asyncio.gather(*tasks)
    except IndexingAborted:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    _save_desc_cache(cache_file, desc_cache)
    logger.info("图片知识图谱摄入完成: %d/%d", done_count, total)
    return done_count
