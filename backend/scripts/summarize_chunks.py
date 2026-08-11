"""Print chunk audit JSON summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(path: str) -> None:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"=== {p} ===")
    print(f"course={data['course_id']} chunks={data['chunk_count']}")
    print(f"files={data['source_files']}")
    print("strategy note: ingest via parse_files + persist_ingest_chunks")
    for i, c in enumerate(data["chunks"]):
        src = data["chunk_sources"][i] if i < len(data.get("chunk_sources", [])) else ""
        preview = c.replace("\n", " ")[:200]
        tag = ""
        if c.count("|") >= 4 and "---" not in c[:30]:
            tag = " [TABLE?]"
        if c.startswith("【章节:"):
            tag += " [HAS_SECTION_PREFIX]"
        print(f"\n--- chunk {i} len={len(c)}{tag}")
        print(f"source: {src}")
        print(preview)


if __name__ == "__main__":
    for fp in sys.argv[1:]:
        summarize(fp)
        print()
