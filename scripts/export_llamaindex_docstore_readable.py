"""将 llamaindex_storage/docstore.json 导出为 UTF-8 Markdown，便于查看全部节点正文。

用法（在仓库根目录）:
  python scripts/export_llamaindex_docstore_readable.py circuit_analysis
  python scripts/export_llamaindex_docstore_readable.py circuit_analysis --out my_export.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根 -> backend
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
KB_ROOT = BACKEND / "data" / "knowledge_bases"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("course_id", help="例如 circuit_analysis")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径（默认: .../llamaindex_storage/docstore_all_readable.md）",
    )
    args = ap.parse_args()

    docstore = KB_ROOT / args.course_id / "llamaindex_storage" / "docstore.json"
    if not docstore.is_file():
        print("找不到:", docstore, file=sys.stderr)
        return 1

    out = args.out or (docstore.parent / "docstore_all_readable.md")

    with open(docstore, encoding="utf-8") as f:
        d = json.load(f)

    data = d.get("docstore/data", {})
    lines = [f"# LlamaIndex docstore 全文导出 — `{args.course_id}`", "", f"节点数: {len(data)}", ""]

    for i, (uid, node) in enumerate(data.items(), 1):
        inner = node.get("__data__", {})
        meta = inner.get("metadata", {}) or {}
        text = inner.get("text", "") or ""
        lines.append("---")
        lines.append(f"## 节点 {i}  `id={uid}`")
        lines.append("")
        for k in sorted(meta.keys()):
            lines.append(f"- **{k}**: {meta.get(k)!r}")
        lines.append("")
        lines.append("### 正文")
        lines.append("")
        lines.append(text)
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print("written:", out.resolve())
    print("size_bytes:", out.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
