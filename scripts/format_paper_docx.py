"""论文 MD -> 期刊排版风格 DOCX。

用法: python scripts/format_paper_docx.py <输入.md> <输出.docx>

做三件事:
1. 基于 pandoc 默认 reference.docx 重写样式: 宋体正文(小四)/黑体标题/
   Times New Roman 西文/全黑配色/三线表/首行缩进2字/1.5倍行距。
2. pandoc 转换。
3. 后处理 document.xml: 题目/作者/单位/图表题居中, 首页注释小五号,
   参考文献悬挂缩进, 图片限宽 14.5cm。

字号规格(Word 投稿版): 中文标题黑体小二18pt / 副标题与一级标题黑体四号14pt /
作者楷体四号14pt / 二级标题黑体小四12pt / 正文宋体小四12pt /
摘要关键词小四(标签黑体) / 图表题五号加粗 / 表内与参考文献五号 /
收稿日期等首页注释小五9pt。
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
    '<w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
    "</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
    '<w:spacing w:after="0" w:line="360" w:lineRule="auto"/>'
    '<w:ind w:firstLineChars="200" w:firstLine="480"/>'
    '<w:jc w:val="both"/>'
    "</w:pPr></w:pPrDefault></w:docDefaults>"
)

HEI = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
    ' w:eastAsia="SimHei"/>'
)


def _heading(sid: str, name: str, sz: int, before: int, after: int, jc: str | None = None) -> str:
    jc_xml = f'<w:jc w:val="{jc}"/>' if jc else ""
    return (
        f'<w:style w:type="paragraph" w:styleId="{sid}">'
        f'<w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
        f'<w:next w:val="BodyText"/>'
        f'<w:pPr><w:keepNext/><w:spacing w:before="{before}" w:after="{after}"/>{jc_xml}'
        f'<w:ind w:firstLine="0" w:firstLineChars="0"/></w:pPr>'
        f"<w:rPr>{HEI}<w:b w:val=\"0\"/><w:sz w:val=\"{sz}\"/><w:szCs w:val=\"{sz}\"/></w:rPr>"
        f"</w:style>"
    )


STYLE_REPLACEMENTS = {
    # 中文标题: 黑体小二 18pt 居中
    "Title": _heading("Title", "Title", 36, 240, 120, jc="center"),
    # 一级标题: 黑体四号 14pt; 二级: 黑体小四 12pt; 三级: 黑体五号 10.5pt
    "Heading1": _heading("Heading1", "heading 1", 28, 240, 120),
    "Heading2": _heading("Heading2", "heading 2", 24, 120, 60),
    "Heading3": _heading("Heading3", "heading 3", 21, 120, 60),
    # 表格单元格: 五号 + 单倍行距 + 不缩进
    "Compact": (
        '<w:style w:type="paragraph" w:styleId="Compact">'
        '<w:name w:val="Compact"/><w:basedOn w:val="Normal"/>'
        '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:firstLine="0" w:firstLineChars="0"/></w:pPr>'
        '<w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
        "</w:style>"
    ),
    # 三线表: 上下粗线 + 表头下细线, 无竖线; 表内五号
    "Table": (
        '<w:style w:type="table" w:styleId="Table">'
        '<w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
        '<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:firstLine="0" w:firstLineChars="0"/></w:pPr>'
        '<w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
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
    styles = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", DOC_DEFAULTS, styles, flags=re.DOTALL)
    for sid, block in STYLE_REPLACEMENTS.items():
        pattern = rf'<w:style [^>]*w:styleId="{sid}"[^>]*>.*?</w:style>'
        styles, n = re.subn(pattern, block, styles, flags=re.DOTALL)
        if n == 0:  # 默认模板缺该样式则追加
            styles = styles.replace("</w:styles>", block + "</w:styles>")
        elif n > 1:
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
    if "hanging" in attrs:  # 悬挂缩进: left+hanging 组合, 不与 firstLine 并存
        vals = {"left": attrs["left"], "hanging": attrs["hanging"], "hangingChars": 200}
    else:  # 默认连字数缩进一起清零(中文版 Word 中 firstLineChars 优先于 firstLine)
        vals = {"firstLine": 0, "firstLineChars": 0}
        vals.update(attrs)
    for k, v in vals.items():
        ind.set(w(k), str(v))
    pPr.append(ind)


def _set_jc(p, val):
    pPr = _pPr(p)
    for old in pPr.findall(w("jc")):
        pPr.remove(old)
    jc = ET.Element(w("jc"))
    jc.set(w("val"), val)
    pPr.append(jc)


def _apply_run_style(r, east=None, sz=None):
    rPr = r.find(w("rPr"))
    if rPr is None:
        rPr = ET.Element(w("rPr"))
        r.insert(0, rPr)
    if east:
        for old in rPr.findall(w("rFonts")):
            rPr.remove(old)
        fonts = ET.Element(w("rFonts"))
        fonts.set(w("ascii"), "Times New Roman")
        fonts.set(w("hAnsi"), "Times New Roman")
        fonts.set(w("eastAsia"), east)
        rPr.append(fonts)
    if sz:
        for tag in ("sz", "szCs"):
            for old in rPr.findall(w(tag)):
                rPr.remove(old)
            el = ET.Element(w(tag))
            el.set(w("val"), str(sz))
            rPr.append(el)


def _style_runs(p, east=None, sz=None, first_only=False):
    if east is None and sz is None:
        return  # 无字体/字号要设时不建空 rPr
    runs = p.findall(w("r"))
    if first_only:
        runs = runs[:1]
    for r in runs:
        _apply_run_style(r, east=east, sz=sz)


def classify(text: str) -> str | None:
    if text.startswith("——"):
        return "subtitle"
    if text.startswith("王浩名") and "游向东" in text:
        return "cn_authors"
    if text.startswith("（1．"):
        return "cn_affil"
    if text.startswith("Design and implementation"):
        return "en_title"
    if text.startswith("WANG Haoming"):
        return "en_authors"
    if text.startswith("(1. "):
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
    "subtitle":    {"jc": "center", "ind": {}, "east": "SimHei", "sz": 28},   # 黑体四号
    "cn_authors":  {"jc": "center", "ind": {}, "east": "KaiTi", "sz": 28},    # 楷体四号
    "cn_affil":    {"jc": "center", "ind": {}, "sz": 24},                     # 宋体小四
    "en_title":    {"jc": "center", "ind": {}, "sz": 28},                     # 四号加粗
    "en_authors":  {"jc": "center", "ind": {}, "sz": 24},                     # 小四
    "en_affil":    {"jc": "center", "ind": {}, "sz": 21},                     # 五号
    "frontmatter": {"jc": "both", "ind": {}, "sz": 24},                       # 小四不缩进
    "footnote":    {"jc": "both", "ind": {}, "sz": 18},                       # 小五(唯一)
    "caption":     {"jc": "center", "ind": {}, "sz": 21},                     # 五号加粗居中
    "reference":   {"jc": "both", "ind": {"left": 420, "hanging": 420}, "sz": 21},  # 五号悬挂
    "formula":     {"jc": "left", "ind": {}},
}


# ---------------------------------------------------------------- 表格重排

PAGE_TEXT_DXA = 9_037  # A4 版心宽 ≈ 15.9 cm


def _units(text: str) -> int:
    """视觉宽度: 中日韩字符计 2, 其余计 1。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _restyle_cell_paragraph(p, align: str):
    """单元格段落显式化: 五号/单倍行距/无缩进/按列对齐(pPr 按 schema 顺序重建)。"""
    pPr = p.find(w("pPr"))
    pStyle = pPr.find(w("pStyle")) if pPr is not None else None
    if pPr is not None:
        for old in list(pPr):
            pPr.remove(old)
    else:
        pPr = ET.Element(w("pPr"))
        p.insert(0, pPr)
    if pStyle is not None:
        pPr.append(pStyle)
    spacing = ET.Element(w("spacing"))
    spacing.set(w("after"), "0")
    spacing.set(w("line"), "240")
    spacing.set(w("lineRule"), "auto")
    pPr.append(spacing)
    ind = ET.Element(w("ind"))
    ind.set(w("firstLine"), "0")
    ind.set(w("firstLineChars"), "0")
    pPr.append(ind)
    jc = ET.Element(w("jc"))
    jc.set(w("val"), align)
    pPr.append(jc)
    for r in p.findall(w("r")):
        _apply_run_style(r, sz=21)


def _restyle_tables(root, counts: dict):
    """pandoc 的 auto 窄表在 Word 里左置且首列易折行: 按内容重算列宽、
    整表居中、固定布局; 表头居中, 窄列居中宽列左对齐。"""
    for tbl in root.iter(w("tbl")):
        rows = tbl.findall(w("tr"))
        grid = tbl.find(w("tblGrid"))
        if not rows or grid is None:
            continue
        cols = grid.findall(w("gridCol"))
        ncols = len(cols)
        if ncols == 0:
            continue
        col_units = [0] * ncols
        for tr in rows:
            for ci, tc in enumerate(tr.findall(w("tc"))[:ncols]):
                text = "".join(t.text or "" for t in tc.iter(w("t")))
                col_units[ci] = max(col_units[ci], _units(text))
        col_dxa = [min(4400, max(900, u * 105 + 230)) for u in col_units]
        total = sum(col_dxa)
        if total > PAGE_TEXT_DXA:  # 超版心按比例收窄
            scale = PAGE_TEXT_DXA / total
            col_dxa = [int(d * scale) for d in col_dxa]
            total = sum(col_dxa)
        for gc, dxa in zip(cols, col_dxa):
            gc.set(w("w"), str(dxa))
        tblPr = tbl.find(w("tblPr"))
        tblW = tblPr.find(w("tblW"))
        tblW.set(w("w"), str(total))
        tblW.set(w("type"), "dxa")
        jc = ET.Element(w("jc"))
        jc.set(w("val"), "center")
        layout = tblPr.find(w("tblLayout"))
        if layout is not None:
            layout.set(w("type"), "fixed")
            tblPr.insert(list(tblPr).index(layout), jc)
        else:
            look = tblPr.find(w("tblLook"))
            pos = list(tblPr).index(look) if look is not None else len(list(tblPr))
            tblPr.insert(pos, jc)
            layout = ET.Element(w("tblLayout"))
            layout.set(w("type"), "fixed")
            tblPr.insert(pos + 1, layout)
        for ri, tr in enumerate(rows):
            for ci, tc in enumerate(tr.findall(w("tc"))[:ncols]):
                tcPr = tc.find(w("tcPr"))
                if tcPr is None:
                    tcPr = ET.Element(w("tcPr"))
                    tc.insert(0, tcPr)
                tcW = tcPr.find(w("tcW"))
                if tcW is None:
                    tcW = ET.Element(w("tcW"))
                    tcPr.insert(0, tcW)
                tcW.set(w("w"), str(col_dxa[ci]))
                tcW.set(w("type"), "dxa")
                if tcPr.find(w("vAlign")) is None:
                    va = ET.Element(w("vAlign"))
                    va.set(w("val"), "center")
                    tcPr.append(va)
                align = "center" if (ri == 0 or col_units[ci] <= 12) else "left"
                for p in tc.findall(w("p")):
                    _restyle_cell_paragraph(p, align)
        counts["tables_restyled"] = counts.get("tables_restyled", 0) + 1


def postprocess_document(doc_xml: bytes) -> tuple[bytes, dict]:
    raw = doc_xml.decode("utf-8")
    for prefix, uri in re.findall(r'xmlns:(\w+)="([^"]+)"', raw):
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(raw)
    counts = {}
    for p in root.iter(w("p")):
        text = _plain_text(p)
        kind = classify(text)
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        act = ACTIONS[kind]
        _set_ind(p, **act["ind"])
        if act.get("jc"):
            _set_jc(p, act["jc"])
        _style_runs(p, east=act.get("east"), sz=act.get("sz"))
        # "摘　要"/"关键词"标签四字用黑体(第一个 run 是加粗标签)
        if kind == "frontmatter" and text.startswith(("摘　要", "关键词")):
            _style_runs(p, east="SimHei", first_only=True)
    # 硬锚点零命中 = 版式将静默降级为正文样式, 必须显式告警
    for required in ("cn_authors", "en_title", "en_authors", "en_affil"):
        if counts.get(required, 0) == 0:
            print(f"WARNING: 段落锚点 {required} 未命中, 对应版式未应用", file=sys.stderr)
    # 图片限宽
    for ext in list(root.iter(f"{{{WP}}}extent")) + list(root.iter(f"{{{A}}}ext")):
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        if cx > MAX_IMG_EMU:
            ratio = MAX_IMG_EMU / cx
            ext.set("cx", str(MAX_IMG_EMU))
            ext.set("cy", str(int(cy * ratio)))
            counts["img_scaled"] = counts.get("img_scaled", 0) + 1
    _restyle_tables(root, counts)
    return ET.tostring(root, xml_declaration=True, encoding="UTF-8"), counts


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("用法: python scripts/format_paper_docx.py <输入.md> <输出.docx>")
    # pathlib 的 / 对绝对路径右操作数直接整体替换, 无需分支
    md, out = ROOT / Path(sys.argv[1]), ROOT / Path(sys.argv[2])

    def _strip_tags(xml: bytes) -> str:
        return re.sub(r"<[^>]+>", "", xml.decode("utf-8"))

    old_text = None
    if out.exists():
        try:
            z = zipfile.ZipFile(out)
            old_text = _strip_tags(z.read("word/document.xml"))
            out.unlink()
        except PermissionError:
            sys.exit(f"输出文件 {out.name} 正被 Word 占用, 请先关闭后重试")

    build_reference()
    subprocess.run(
        ["pandoc", str(md), "-o", str(out),
         f"--reference-doc={REFERENCE_DOCX}", "--resource-path=.",
         # MD 的 # 是文题而非章节: 升格为 docx Title, ## -> Heading1, ### -> Heading2
         "--shift-heading-level-by=-1"],
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
        print("正文内容与旧版一致:", old_text == _strip_tags(new_doc))
    print("输出:", out.resolve(), f"({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
