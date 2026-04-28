"""项目架构.md -> 项目架构.pdf（Chrome 无头打印，排版向印刷品靠拢）。"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "项目架构.md"
HTML_OUT = ROOT / "_项目架构_for_pdf.html"
PDF_OUT = ROOT / "项目架构.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

PRINT_CSS = """
/* ----- 页面与容器 ----- */
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
}

* {
  box-sizing: border-box;
}

html {
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

body {
  margin: 0;
  padding: 0;
  font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
    "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  line-height: 1.58;
  color: #1a202c;
  background: #fff;
}

article.pdf-doc {
  max-width: 100%;
}

/* ----- 文首主标题 ----- */
article.pdf-doc > h1:first-of-type {
  font-size: 22pt;
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.02em;
  margin: 0 0 0.35em 0;
  padding-bottom: 0.45em;
  border-bottom: 3px solid #2c5282;
  color: #1a365d;
  page-break-after: avoid;
}

/* ----- 二级标题（章节） ----- */
h2 {
  font-size: 14.5pt;
  font-weight: 700;
  color: #2c5282;
  margin: 1.35em 0 0.55em 0;
  padding: 0 0 0.28em 0;
  border-bottom: 1px solid #bee3f8;
  page-break-after: avoid;
  break-after: avoid-page;
}

h2:first-of-type {
  margin-top: 0.85em;
}

/* ----- 三级标题 ----- */
h3 {
  font-size: 11.5pt;
  font-weight: 600;
  color: #2d3748;
  margin: 1.05em 0 0.4em 0;
  page-break-after: avoid;
}

h4, h5, h6 {
  font-size: 10.5pt;
  font-weight: 600;
  margin: 0.9em 0 0.35em 0;
  page-break-after: avoid;
}

/* ----- 段落与列表 ----- */
p {
  margin: 0.5em 0;
  orphans: 3;
  widows: 3;
}

ul, ol {
  margin: 0.45em 0 0.65em 0;
  padding-left: 1.45em;
}

li {
  margin: 0.22em 0;
}

li > p {
  margin: 0.28em 0;
}

/* ----- 文首摘要引用 ----- */
blockquote {
  margin: 0.85em 0;
  padding: 0.65em 1em 0.65em 1.1em;
  border-left: 4px solid #63b3ed;
  background: #ebf8ff;
  color: #2c5282;
  font-size: 10pt;
  line-height: 1.5;
}

blockquote p {
  margin: 0.35em 0;
}

/* ----- 分隔线 ----- */
hr {
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 1.25em 0;
}

/* ----- 行内代码 ----- */
code {
  font-family: "Cascadia Code", "Consolas", "Microsoft YaHei UI", monospace;
  font-size: 0.88em;
  background: #edf2f7;
  color: #2d3748;
  padding: 0.12em 0.38em;
  border-radius: 3px;
  border: 1px solid #e2e8f0;
}

/* ----- 代码块 / ASCII 架构图 ----- */
pre {
  font-family: "Cascadia Code", "Consolas", "Microsoft YaHei UI", monospace;
  font-size: 8.35pt;
  line-height: 1.42;
  background: #f7fafc;
  border: 1px solid #cbd5e0;
  border-radius: 4px;
  padding: 10px 12px;
  margin: 0.75em 0;
  white-space: pre-wrap;
  word-break: break-all;
  overflow-wrap: anywhere;
  page-break-inside: auto;
}

pre code {
  background: transparent;
  border: none;
  padding: 0;
  font-size: inherit;
  color: #1a202c;
}

/* ----- 表格 ----- */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.75em 0 1em 0;
  font-size: 9.25pt;
  border: 1px solid #a0aec0;
  page-break-inside: auto;
}

thead {
  display: table-header-group;
}

tr {
  page-break-inside: avoid;
  break-inside: avoid;
}

th, td {
  border: 1px solid #cbd5e0;
  padding: 6px 9px;
  vertical-align: top;
  text-align: left;
  word-break: break-word;
}

th {
  background: #2c5282;
  color: #fff;
  font-weight: 600;
}

tbody tr:nth-child(even) td {
  background: #f7fafc;
}

tbody tr:nth-child(odd) td {
  background: #fff;
}

/* ----- 图片 ----- */
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1em auto;
  page-break-inside: avoid;
  page-break-before: auto;
}

/* ----- 目录与锚点链接 ----- */
a {
  color: #2b6cb0;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

/* ----- 强调 ----- */
strong {
  font-weight: 700;
  color: #1a202c;
}
"""


def main() -> int:
    import markdown

    text = MD.read_text(encoding="utf-8")

    def fix_img(m: re.Match[str]) -> str:
        alt, raw = m.group(1), m.group(2).strip()
        p = Path(raw)
        try:
            if p.is_file():
                return f"![{alt}]({p.as_uri()})"
        except OSError:
            pass
        return m.group(0)

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_img, text)
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    css = textwrap.dedent(PRINT_CSS).strip()
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>课程Agent — 项目架构</title>
<style>
{css}
</style>
</head>
<body>
<article class="pdf-doc">
{body}
</article>
</body>
</html>"""
    HTML_OUT.write_text(doc, encoding="utf-8")

    if not CHROME.is_file():
        print("Chrome not found at", CHROME, file=sys.stderr)
        return 1
    subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_OUT}",
            HTML_OUT.as_uri(),
        ],
        check=True,
    )
    print("Wrote", PDF_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
