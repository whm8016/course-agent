# -*- coding: utf-8 -*-
"""论文 MD -> 期刊排版风格 DOCX。

用法: python scripts/format_paper_docx.py <输入.md> <输出.docx>

做三件事:
1. 基于 pandoc 默认 reference.docx 重写样式: 宋体正文(五号)/黑体标题/
   Times New Roman 西文/全黑配色/三线表/首行缩进2字/1.5倍行距。
2. pandoc 转换。
3. 后处理 document.xml: 题目/作者/单位/图表题居中, 首页注释与参考文献
   小五号, 参考文献悬挂缩进, 图片限宽 14.5cm。
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DOCX = Path(__file__).resolve().parent / "paper_reference.docx"

MAX_IMG_EMU = 5_220_000  # 14.5 cm

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


# ---------------------------------------------------------------- 样式模板

DOC_DEFAULTS = (
    "<w:docDefaults><w:rPrDefault><w:rPr>"
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
    ' w:eastAsia="SimSun" w:cs="Times New Roman"/>'
    '<w:color w:val="000000"/><w:sz w:val="21"/><w:szCs w:val="21"/>'
    "</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
    '<w:spacing w:after="0" w:line="360" w:lineRule="auto"/>'
    '<w:ind w:firstLineChars="200" w:firstLine="420"/>'
    '<w:jc w:val="both"/>'
    "</w:pPr></w:pPrDefault></w:docDefaults>"
)

HEI = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
    ' w:eastAsia="SimHei"/>'
)

STYLE_REPLACEMENTS = {
    "Title": (
        '<w:style w:type="paragraph" w:styleId="Title">'
        '<w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="BodyText"/>'
        '<w:pPr><w:spacing w:before="240" w:after="120"/><w:jc w:val="center"/>'
        '<w:ind w:firstLine="0"/></w:pPr>'
        f"<w:rPr>{HEI}<w:b w:val=\"0\"/><w:sz w:val=\"32\"/><w:szCs w:val=\"32\"/></w:rPr>"
        "</w:style>"
    ),
    "Heading1": (
        '<w:style w:type="paragraph" w:styleId="Heading1">'
        '<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
        '<w:next w:val="BodyText"/>'
        '<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/>'
        '<w:ind w:firstLine="0"/></w:pPr>'
        f"<w:rPr>{HEI}<w:b w:val=\"0\"/><w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/></w:rPr>"
        "</w:style>"
    ),
    "Heading2": (
        '<w:style w:type="paragraph" w:styleId="Heading2">'
        '<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>'
        '<w:next w:val="BodyText"/>'
        '<w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/>'
        '<w:ind w:firstLine="0"/></w:pPr>'
        f"<w:rPr>{HEI}<w:b w:val=\"0\"/><w:sz w:val=\"21\"/><w:szCs w:val=\"21\"/></w:rPr>"
        "</w:style>"
    ),
    # 表格单元格: 小五号 + 单倍行距 + 不缩进
    "Compact": (
        '<w:style w:type="paragraph" w:styleId="Compact">'
        '<w:name w:val="Compact"/><w:basedOn w:val="Normal"/>'
        '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:firstLine="0"/></w:pPr>'
        '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        "</w:style>"
    ),
    # 三线表: 上下粗线 + 表头下细线, 无竖线
    "Table": (
        '<w:style w:type="table" w:styleId="Table">'
        '<w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
        '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:firstLine="0"/></w:pPr>'
        '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        "<w:tblPr><w:tblBorders>"
        '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
        '<w:tblCellMar><w:top w:w="28" w:type="dxa"/><w:left w:w="85" w:type="dxa"/>'
        '<w:bottom w:w="28" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
        '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr><w:tcBorders>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        "</w:tcBorders></w:tcPr></w:tblStylePr>"
        "</w:style>"
    ),
}


def build_reference() -> Path:
    """从 pandoc 默认模板构造期刊风格 reference.docx。"""
    raw = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.docx"],
        capture_output=True, check=True,
    ).stdout
    zin = zipfile.ZipFile(io.BytesIO(raw))
    styles = zin.read("word/styles.xml").decode("utf-8")
    styles = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", DOC_DEFAULTS, styles, flags=re.S)
    for sid, block in STYLE_REPLACEMENTS.items():
        pattern = r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % sid
        styles, n = re.subn(pattern, block, styles, flags=re.S)
        if n != 1:
            raise RuntimeError(f"reference 样式 {sid} 替换异常: {n} 处")
    with zipfile.ZipFile(REFERENCE_DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = styles.encode("utf-8") if item.filename == "word/styles.xml" else zin.read(item.filename)
            zout.writestr(item.filename, data)
    return REFERENCE_DOCX


# ---------------------------------------------------------------- 后处理

def _plain_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(w("t")))


def _pPr(p):
    el = p.find(w("pPr"))
    if el is None:
        el = ET.Element(w("pPr"))
        p.insert(0, el)
    return el


def _set_ind(p, **attrs):
    pPr = _pPr(p)
    for old in pPr.findall(w("ind")):
        pPr.remove(old)
    ind = ET.Element(w("ind"))
    for k, v in attrs.items():
        ind.set(w(k), str(v))
    pPr.append(ind)


def _set_jc(p, val):
    pPr = _pPr(p)
    for old in pPr.findall(w("jc")):
        pPr.remove(old)
    jc = ET.Element(w("jc"))
    jc.set(w("val"), val)
    pPr.append(jc)


def _style_runs(p, east=None, ascii_=None, sz=None):
    """给段落内所有 run 设置字体/字号(半磅)。"""
    for r in p.findall(w("r")):
        rPr = r.find(w("rPr"))
        if rPr is None:
            rPr = ET.Element(w("rPr"))
            r.insert(0, rPr)
        if east or ascii_:
            for old in rPr.findall(w("rFonts")):
                rPr.remove(old)
            fonts = ET.Element(w("rFonts"))
            fonts.set(w("ascii"), ascii_ or "Times New Roman")
            fonts.set(w("hAnsi"), ascii_ or "Times New Roman")
            fonts.set(w("eastAsia"), east or "SimSun")
            rPr.append(fonts)
        if sz:
            for tag in ("sz", "szCs"):
                for old in rPr.findall(w(tag)):
                    rPr.remove(old)
            for tag in ("sz", "szCs"):
                el = ET.Element(w(tag))
                el.set(w("val"), str(sz))
                rPr.append(el)


def classify(text: str) -> str | None:
    if text.startswith("——"):
        return "subtitle"
    if text.startswith("王浩名") and "游向东" in text:
        return "cn_authors"
    if text.startswith("（1．"):
        return "cn_affil"
    if text.startswith("Design and Implementation"):
        return "en_title"
    if text.startswith("WANG Haoming"):
        return "en_authors"
    if text.startswith("(1. School of Information"):
        return "en_affil"
    if text.startswith(("摘　要", "关键词", "中图分类号", "Abstract", "Keywords")):
        return "frontmatter"
    if text.startswith(("收稿日期", "作者简介", "通信作者")):
        return "footnote"
    if re.match(r"^(图|表)[ 　]?\d", text):
        return "caption"
    if re.match(r"^\[\d+\]", text):
        return "reference"
    if text.startswith("　　"):
        return "formula"
    return None


ACTIONS = {
    "subtitle":   dict(jc="center", ind={}, east="SimHei", sz=28),
    "cn_authors": dict(jc="center", ind={}, east="KaiTi", sz=24),
    "cn_affil":   dict(jc="center", ind={}, sz=18),
    "en_title":   dict(jc="center", ind={}, sz=24),
    "en_authors": dict(jc="center", ind={}, sz=21),
    "en_affil":   dict(jc="center", ind={}, sz=18),
    "frontmatter": dict(jc="both", ind={}, sz=18),
    "footnote":   dict(jc="both", ind={}, sz=18),
    "caption":    dict(jc="center", ind={}, sz=18),
    "reference":  dict(jc="both", ind={"left": 420, "hanging": 420}, sz=18),
    "formula":    dict(jc="left", ind={}),
}


def postprocess_document(doc_xml: bytes) -> tuple[bytes, dict]:
    raw = doc_xml.decode("utf-8")
    for prefix, uri in re.findall(r'xmlns:(\w+)="([^"]+)"', raw):
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(raw)
    counts = {}
    for p in root.iter(w("p")):
        kind = classify(_plain_text(p))
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        act = ACTIONS[kind]
        if act.get("jc"):
            _set_jc(p, act["jc"])
        if "ind" in act:
            _set_ind(p, **act["ind"])
        _style_runs(p, east=act.get("east"), sz=act.get("sz"))
    # 图片限宽
    scaled = 0
    for ext in list(root.iter(f"{{{WP}}}extent")) + list(root.iter(f"{{{A}}}ext")):
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        if cx > MAX_IMG_EMU:
            ratio = MAX_IMG_EMU / cx
            ext.set("cx", str(MAX_IMG_EMU))
            ext.set("cy", str(int(cy * ratio)))
            scaled += 1
    return ET.tostring(root, xml_declaration=True, encoding="UTF-8"), counts


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("用法: python scripts/format_paper_docx.py <输入.md> <输出.docx>")
    md, out = Path(sys.argv[1]), Path(sys.argv[2])
    md = md if md.is_absolute() else ROOT / md
    out = out if out.is_absolute() else ROOT / out

    old_text = None
    if out.exists():
        try:
            z = zipfile.ZipFile(out)
            old_text = re.sub(
                r"<[^>]+>", "", z.read("word/document.xml").decode("utf-8")
            )
            out.unlink()
        except PermissionError:
            sys.exit(f"输出文件 {out.name} 正被 Word 占用, 请先关闭后重试")

    build_reference()
    subprocess.run(
        ["pandoc", str(md), "-o", str(out),
         f"--reference-doc={REFERENCE_DOCX}", "--resource-path=."],
        cwd=ROOT, check=True,
    )

    zin = zipfile.ZipFile(out)
    new_doc, counts = postprocess_document(zin.read("word/document.xml"))
    tmp = out.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = new_doc if item.filename == "word/document.xml" else zin.read(item.filename)
            zout.writestr(item.filename, data)
    zin.close()
    os.replace(tmp, out)

    print("段落处理统计:", counts)
    if old_text is not None:
        new_text = re.sub(r"<[^>]+>", "", new_doc.decode("utf-8"))
        print("正文内容与旧版一致:", old_text == new_text)
    print("输出:", out.resolve(), f"({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
