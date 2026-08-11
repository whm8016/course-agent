"""markdown → sections（DeepTutor 式「解析归解析、切块归切块」路线）。

MinerU 的 ``full.md`` 实测已是干净、结构完整的 markdown（header/page_footnote/aside_text
0 条混入；标题层级写进 ``#``/``##``；图片 ``![](images/<sha>.jpg)`` 后 78% 紧跟图注）。
本模块用 LlamaIndex ``MarkdownNodeParser`` 按标题切节，产出与 ``extract_pdf_sections``
同构的 ``[{title, content, page}]``；**大小仍交给下游 ingestion 的 SentenceSplitter**——
不在解析层预写 chunk（与 DeepTutor ``document_loader.py`` 一致）。

为什么走 markdown 而非继续修 ``blocks_to_sections``：MinerU 标题块是 ``type=text`` 带
``text_level``（非 ``type=title``），``blocks_to_sections`` 分不出节，会把整篇 PDF 拼成
一个巨 section。markdown 路线绕开 blocks 直接吃 ``full.md`` 的标题层级——这才是 MinerU
PDF 的正确分节方式，且 ``ParsedDocument.markdown`` 是 engine-agnostic IR 的最低公约数。
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 节点内容首行的叶子标题：## 6.1 Main Results / # Title / ### References
_LEAF_HEADER_RE = re.compile(r"^\s*#{1,6}\s+(.*?)\s*$", re.MULTILINE)
# 图片 marker：![](images/<name>)（MinerU 产物，alt 通常为空）
_IMAGE_MARKER_RE = re.compile(r"!\[[^\]]*\]\(images/([^)]+)\)")
# 图注首行特征——marker 后紧跟这类句才算「已有图注」，不调 VLM
_CAPTION_HINT_RE = re.compile(r"^(Figure|Fig\.?|Table|Tab\.?|图|表)\b", re.IGNORECASE)
# 参考文献章节标题（References / 参考文献 / Bibliography / Works Cited ...）
_REF_SECTION_RE = re.compile(
    r"^(references|bibliography|works\s*cited|参考文献|引用文献|文献)\b",
    re.IGNORECASE,
)
# 子图注误标标题：MinerU 偶尔把「(a)/(b)/(c) ...」这类子图/子表注漏挂到 image_caption，
# 单独识别成 text_level 标题（同一文档里其余子图注常常又挂对了，行为不一致）。
# 这类"标题"不代表真实小节边界，会把配图/表格从其所属小节里切散——识别后并回上一节。
_SUBFIGURE_CAPTION_RE = re.compile(r"^[\(（][a-zA-Z0-9][\)）]\s")
# VLM 图片描述 prompt：_vlm_caption_sync（单张串行）与 _prefetch_descriptions（并发预取）
# 共用 image_extractor 的模块级常量（DRY，避免两处漂移导致描述风格不一致）。


def _is_reference_section(leaf_title: str) -> bool:
    """叶子标题是否属于参考文献章节。"""
    return bool(leaf_title and _REF_SECTION_RE.match(leaf_title.strip()))


def _heading_page_index(blocks: Optional[list[dict[str, Any]]]) -> dict[str, int]:
    """从 MinerU ``content_list`` 建「标题文本 → page_idx」查找表，供 section 回填页码。

    MinerU 标题块是 ``type=text`` 带 ``text_level``（1-6）；docling 产 ``type=title``。
    两者都收。同名标题取首次出现的页码（论文里章节标题一般唯一）。
    """
    page_of: dict[str, int] = {}
    if not blocks:
        return page_of
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        btype = str(blk.get("type") or "").lower()
        if "text_level" not in blk and btype not in ("title", "section_header"):
            continue
        text = str(blk.get("text") or "").strip()
        if not text or text in page_of:
            continue
        page_of[text] = int(blk.get("page_idx") or blk.get("page") or 0)
    return page_of


def markdown_to_sections(
    markdown: str,
    *,
    blocks: Optional[list[dict[str, Any]]] = None,
    drop_refs: bool = True,
) -> list[dict]:
    """``full.md`` → sections（``[{title, content, page}]``，与 ``extract_pdf_sections`` 同构）。

    ``MarkdownNodeParser`` 只按标题切节、不控大小（大小交给下游 SentenceSplitter）。
    注意其 ``header_path`` 元数据**只含祖先标题**，叶子标题是该节点内容的首行 ``#``——
    故 ``title`` 与参考文献过滤都从内容首行取叶子，而非 ``header_path``。

    ``title`` 取**叶子标题**（如「6.1 Main Results」），与 DOCX/PPTX/docling-PDF 各
    extractor 一致（均叶子标题），避免论文长标题污染每个 chunk 的【章节:…】来源前缀。
    """
    if not markdown or not markdown.strip():
        return []
    from llama_index.core.node_parser import MarkdownNodeParser  # noqa: PLC0415
    from llama_index.core.schema import Document as _LIDoc  # noqa: PLC0415

    nodes = MarkdownNodeParser().get_nodes_from_documents([_LIDoc(text=markdown)])
    page_of = _heading_page_index(blocks)

    out: list[dict] = []
    for n in nodes:
        text = n.get_content().strip()
        if not text:
            continue
        m = _LEAF_HEADER_RE.match(text)
        leaf = m.group(1).strip() if m else ""
        if drop_refs and _is_reference_section(leaf):
            continue
        if out and _SUBFIGURE_CAPTION_RE.match(leaf):
            # 误标的子图注：去掉 "#" 前缀还原成正文，并入上一节而非另起一节。
            lines = text.split("\n", 1)
            body = leaf if len(lines) == 1 else f"{leaf}\n{lines[1]}"
            out[-1]["content"] = f"{out[-1]['content']}\n\n{body}"
            continue
        out.append({"title": leaf, "content": text, "page": page_of.get(leaf, 0)})
    return out


# ── 图片引用 inline（pgvector 无多模态 embedding，图片信息必须以文本进 markdown）────


def _describe_image(
    img_name: str,
    asset_dir: Optional[Path],
    desc_cache: dict[str, str],
) -> tuple[str, bool]:
    """fallback：给无图注的图片一个文本描述（命中缓存→VLM→都失败返回 ''）。

    ``asset_dir`` 是 ``images/`` 目录；marker 里的 ``images/<name>`` 即 ``asset_dir/<name>``。
    desc_cache 按图片**字节** sha256 记账（与 image_extractor 的 base64-串 key 不互通——故
    ``_vlm_caption_sync`` 给 caption_func 传空 dict 关闭其内部缓存，由本函数单点管）。
    返回 ``(描述, 是否新写入缓存)``，供调用方按需落盘（命中缓存不触发写）。
    """
    if not asset_dir:
        return "", False
    path = asset_dir / img_name
    if not path.is_file():
        return "", False
    try:
        img_bytes = path.read_bytes()  # 只读一次：算 key + base64 共用
    except OSError:
        return "", False
    key = hashlib.sha256(img_bytes).hexdigest()
    if key in desc_cache:
        return desc_cache[key], False
    desc = _vlm_caption_sync(img_bytes)
    if desc:
        desc_cache[key] = desc
        return desc, True
    return "", False


def _vlm_caption_sync(img_bytes: bytes) -> str:
    """调 vision 模型描述图片（同步封装 image_extractor 的 async caption）。失败返回 ''。

    opt-in：需配 ``settings.vision``（api_key/model）+ DashScope 网络。复用
    ``_make_vision_caption_func`` 的 DashScope 调用逻辑（DRY），缓存交由 ``_describe_image``
    单点管理（caption_func 内部缓存传空 dict 关闭——其 key 是 base64-串 sha256，与外层
    raw-bytes key 不互通）。``asyncio.run`` 安全——本函数仅在摄入线程（``parse_files`` 经
    ``asyncio.to_thread``）内调用，该线程无运行中的 loop；检测到已在 loop 内（如某些 async
    测试）则降级跳过，不阻塞。
    """
    try:
        import asyncio
        import base64

        from core.rag.llamaindex.image_extractor import (
            _IMAGE_DESC_PROMPT,
            _make_vision_caption_func,
        )
    except ImportError:  # 可选依赖未装/未配置，降级
        return ""
    try:
        asyncio.get_running_loop()
        return ""  # 已在事件循环内，run 会爆——降级
    except RuntimeError:
        pass
    try:
        image_data = base64.b64encode(img_bytes).decode()
        caption_func = _make_vision_caption_func({}, asyncio.Lock())
        desc = asyncio.run(
            caption_func(_IMAGE_DESC_PROMPT, image_data=image_data)
        )
        return (desc or "").strip()
    except Exception as exc:  # noqa: BLE001 — VLM 失败不阻断摄入
        logger.debug("inline 图片 VLM 描述失败: %s", exc)
        return ""


def _prefetch_descriptions(
    names: list[str],
    asset_dir: Path,
    desc_cache: dict[str, str],
) -> bool:
    """并发预取未缓存图片的 VLM 描述，写入 ``desc_cache``，返回是否有新写入。

    解决 ``vlm_always`` 模式下 181 张图串行 ``_vlm_caption_sync`` 的十分钟级耗时：每张各自
    ``asyncio.run`` + 新建 ``AsyncOpenAI`` client（见 ``_vlm_caption_sync``）。这里一次
    ``asyncio.run`` 内 ``asyncio.gather`` + ``Semaphore(image_ingest.semaphore)`` 并发预取，
    复用 ``_make_vision_caption_func`` 的 DashScope 调用。按图片**字节** sha256 去重，264 个
    文件里的重复图只调一次。

    单张失败只记 debug 日志、不中断整批（与 ``_vlm_caption_sync`` 的「VLM 失败不阻断摄入」一致）。
    """
    import asyncio
    import base64

    # 1. 按 sha256(bytes) 去重收集 pending：跳过已缓存、读不到的。dict.fromkeys 先对同名 marker
    #    去重（同一图被引用 N 次），内容级去重再靠 pending 的 key 收口。
    pending: dict[str, str] = {}  # key=sha256(bytes) → base64
    for name in dict.fromkeys(names):
        path = asset_dir / name
        try:
            img_bytes = path.read_bytes()
        except OSError:
            continue
        key = hashlib.sha256(img_bytes).hexdigest()
        if key in desc_cache or key in pending:
            continue
        pending[key] = base64.b64encode(img_bytes).decode()

    if not pending:
        return False

    # 2. 已在事件循环内则降级（沿用 _vlm_caption_sync 的判据）——预取是优化，不可阻塞 async 上下文
    try:
        asyncio.get_running_loop()
        return False
    except RuntimeError:
        pass

    # 3. 一次 asyncio.run + gather + Semaphore。caption_func 内部缓存传空 dict 关闭（其 key 是
    #    base64-串 sha256，与外层 raw-bytes key 不互通，desc_cache 由本函数单点管）。
    #    _SEMAPHORE_LIMIT 复用 image_extractor 模块级常量，与 ingest_images_from_files 同口径
    #    （避免两处各自读 settings 漂移）。
    try:
        from core.rag.llamaindex.image_extractor import (  # noqa: PLC0415
            _IMAGE_DESC_PROMPT,
            _SEMAPHORE_LIMIT,
            _make_vision_caption_func,
        )
    except ImportError:  # 可选依赖未装/未配置，降级
        return False

    async def _run() -> int:
        caption_func = _make_vision_caption_func({}, asyncio.Lock())
        sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        async def _one(key: str, image_data: str) -> bool:
            async with sem:
                try:
                    desc = await caption_func(_IMAGE_DESC_PROMPT, image_data=image_data)
                except Exception as exc:  # noqa: BLE001 — 单张失败不中断整批
                    logger.debug("预取图片 VLM 描述失败 %s: %s", key[:8], exc)
                    return False
            desc = (desc or "").strip()
            if desc:
                desc_cache[key] = desc
                return True
            return False

        return sum(await asyncio.gather(*(_one(k, b) for k, b in pending.items())))

    try:
        wrote = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — 预取整体失败不阻断摄入，退回 _replace 内串行 _describe_image
        logger.debug("预取图片描述整体失败: %s", exc)
        return False
    return wrote > 0


def resolve_image_refs(
    markdown: str,
    asset_dir: Optional[Path] = None,
    *,
    vlm_always: bool | None = None,
) -> str:
    """处理 markdown 里的 ``![](images/...)``：inline 成文本（pgvector 无多模态 embedding）。

    - 默认（marker 后紧跟 ``Figure/表/图`` 图注，MinerU 已写入 markdown）：删 marker、留
      图注，**不调 VLM**。
    - fallback（marker 后无图注）：查 ``asset_dir`` 定位文件 → 复用 ``image_extractor`` 的
      desc_cache / VLM caption → 替换为 ``[图: 描述]``；仍失败 → ``[图: 文件名]``，不抛异常。
    - ``vlm_always``（默认读 ``settings.parsing.image_vlm_always``）：开启后有图注的图片也调
      VLM，在 marker 位置插入 ``[图: 描述]``，图注仍是原文下一行——描述补结构/数字、图注保留
      作者撰写的权威命名，两者互补。此时先 ``_prefetch_descriptions`` 并发预取（181 张图不再
      串行 10 分钟），之后 ``re.sub`` 全是缓存命中。

    不调 VLM 是常态，不是缺失——4G + 多租户下 VLM 是 opt-in fallback。
    """
    if not markdown or "](images/" not in markdown:
        return markdown

    from core.rag.llamaindex.image_extractor import (  # noqa: PLC0415
        _load_desc_cache,
        _save_desc_cache,
    )

    if vlm_always is None:
        from settings import get_settings  # noqa: PLC0415

        vlm_always = get_settings().parsing.image_vlm_always

    cache_path = (asset_dir / "desc_cache.json") if asset_dir else None
    # 懒加载：仅命中「需描述」marker 才读 desc_cache；图都有图注且非 always（常态）则零 JSON I/O。
    # dirty：只在缓存新写入时落盘，命中已有条目不触发写。
    desc_cache: dict[str, str] | None = None
    cache_dirty = False

    # always 模式 + 有 asset_dir：先并发预取未缓存图片，灌进 desc_cache，后续 _replace 全命中缓存。
    if vlm_always and asset_dir:
        names = _IMAGE_MARKER_RE.findall(markdown)
        desc_cache = _load_desc_cache(cache_path)
        if _prefetch_descriptions(names, asset_dir, desc_cache):
            cache_dirty = True

    def _replace(m: re.Match[str]) -> str:
        nonlocal desc_cache, cache_dirty
        has_caption = bool(_CAPTION_HINT_RE.match(markdown[m.end():].lstrip()))
        if has_caption and not vlm_always:
            return ""  # 删 marker，图注原文保留（现有行为）
        if desc_cache is None:
            desc_cache = _load_desc_cache(cache_path) if cache_path else {}
        name = m.group(1)
        desc, wrote = _describe_image(name, asset_dir, desc_cache)
        if wrote:
            cache_dirty = True
        if desc:
            return f"[图: {desc}]"  # 有图注时描述在前、图注原文紧随其后
        return "" if has_caption else f"[图: {Path(name).stem}]"

    new_md = _IMAGE_MARKER_RE.sub(_replace, markdown)
    if cache_path and cache_dirty:
        _save_desc_cache(cache_path, desc_cache)
    return new_md


__all__ = [
    "markdown_to_sections",
    "resolve_image_refs",
    "_heading_page_index",
    "_is_reference_section",
]
