"""CitationManager — 引用去重 + 编号 + 渲染（citation_manager）。

基础版：用自增整数编号（``[1]``、``[2]``）而非 的 ``CIT-x-yy``。
按 url / title / source 去重；render_references() 出 Markdown 有序列表附录，
inline_marker(source) 出 ``[n]`` 供报告正文行内引用。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    """一条引用：去重 key + 编号 + 展示文本。"""

    number: int                    # 自增编号（从 1 开始）
    url: str = ""                  # 来源链接（web_search 优先）
    title: str = ""                # 来源标题（论文 / 网页 / 文件名）
    tool_type: str = ""            # rag / web_search / ...
    query: str = ""                # 产生该来源的检索词
    snippet: str = ""              # 简短摘要（可选）

    def dedup_key(self) -> str:
        """去重键：url 优先，其次归一化 title，最后 query。空则用 snippet 兜底。"""
        if self.url:
            return f"url:{self.url.strip()}"
        title = " ".join(self.title.split()).lower()
        if title:
            return f"title:{title}"
        query = " ".join(self.query.split()).lower()
        if query:
            return f"query:{query}"
        return f"snippet:{self.snippet[:80]}"

    def display(self) -> str:
        """渲染为 Markdown 列表项文本（不含编号前缀）。"""
        label = self.title.strip() or self.url.strip() or self.query.strip()
        tool = self.tool_type.replace("_", " ").title()
        parts: list[str] = []
        if label:
            if self.url and self.url != label:
                parts.append(f"[{label}]({self.url})")
            else:
                parts.append(label)
        elif self.url:
            parts.append(self.url)
        meta_bits: list[str] = []
        if tool:
            meta_bits.append(tool)
        if self.query and self.query.strip() != label:
            meta_bits.append(f"query: {self.query.strip()}")
        if meta_bits:
            parts.append("（" + "；".join(meta_bits) + "）")
        return "".join(parts) if parts else "(来源)"


class CitationManager:
    """引用注册中心：去重、编号、渲染附录。

    用法：
        cm = CitationManager()
        cid = cm.add_source(url=..., title=..., tool_type="web_search", query=...)
        marker = cm.inline_marker(cid)          # "[3]"
        appendix = cm.render_references()       # "## 参考资料\\n1. ...\\n2. ..."
    """

    def __init__(self) -> None:
        self._by_key: dict[str, Citation] = {}
        self._by_id: dict[int, Citation] = {}
        self._counter = 0

    def add_source(
        self,
        *,
        url: str = "",
        title: str = "",
        tool_type: str = "",
        query: str = "",
        snippet: str = "",
    ) -> int:
        """登记一条来源，返回其编号（已存在则返回既有编号，不新增）。"""
        cand = Citation(
            number=0,
            url=url,
            title=title,
            tool_type=tool_type,
            query=query,
            snippet=snippet,
        )
        key = cand.dedup_key()
        existing = self._by_key.get(key)
        if existing is not None:
            # 合并补全：旧条目缺 title/url 时用新信息补上
            if not existing.title and title:
                existing.title = title
            if not existing.url and url:
                existing.url = url
            if not existing.snippet and snippet:
                existing.snippet = snippet
            return existing.number
        self._counter += 1
        cand.number = self._counter
        self._by_key[key] = cand
        self._by_id[cand.number] = cand
        return cand.number

    def inline_marker(self, citation_id: int) -> str:
        """返回正文行内引用标记 ``[n]``；未知编号返回空串。"""
        if citation_id in self._by_id:
            return f"[{citation_id}]"
        return ""

    def get(self, citation_id: int) -> Citation | None:
        return self._by_id.get(citation_id)

    def all_citations(self) -> list[Citation]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def __len__(self) -> int:
        return len(self._by_id)

    def render_references(self, heading: str = "参考资料") -> str:
        """渲染 Markdown 有序列表附录。无引用时返回空串。"""
        items = self.all_citations()
        if not items:
            return ""
        lines = [f"## {heading}", ""]
        for c in items:
            lines.append(f"{c.number}. {c.display()}")
        return "\n".join(lines)


__all__ = ["Citation", "CitationManager"]
