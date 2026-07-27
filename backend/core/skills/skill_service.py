"""SkillService — SKILL.md 知识包的三层存储与运行时加载（式渐进式揭示）。

三层（读取优先级 personal > course > builtin）：
- personal: data/skills_user/<user_id>/  —— 用户私人 playbook（多租户隔离，学生侧）
- course  : data/skills/<course_id or _global>/ —— 教师为课程统一制定，同课共享
- builtin : core/skills/builtin/  —— 随包发布，只读

写入层：personal_root 存在 → 写 personal；否则写 course（教师场景保持原状）。
同名时 personal 覆盖 course 的可见性（学生可用私人版覆盖课程版）。

系统提示词只放每个 skill 的单行清单（``render_skills_manifest``），完整内容由
``read_skill`` 工具在任务匹配时拉取（避免 prompt 膨胀）。``always: true`` 的 skill
全文急切注入（用于每轮都适用的全局守则）。


保留 user 层 CRUD（create/update/delete）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from settings import BASE_DIR

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAX_READ_CHARS = 100_000

# manifest 预算（对齐 Anthropic Agent Skills「name+description 约 100 词/skill、总量控制」原则）。
# skill 数量增长时避免清单线性膨胀撑爆 system prompt；超限时按 personal>course>builtin 的既有
# 优先级从尾部裁剪。不在此做语义相关性排序——那需要 query+embedding，属检索层职责（plan 第三条
# 明确：skill 规模未到二三十个前不引入检索层，当前靠优先级截断即可）。
_MANIFEST_MAX_ENTRIES = 40      # 最多列出多少个 skill 行
_MANIFEST_MAX_CHARS = 6_000     # 整块 manifest（含标题）字符预算
_MANIFEST_DESC_MAX_CHARS = 200  # 单条 description 截断长度（超长描述收省略号）

# always skill 全文注入预算（load_for_context，每轮 system prompt 携带）。比 manifest 略宽
# （always 是核心守则），但仍需上限——原先每个 always:true skill 裸 read_text 拼接无限制，
# 多个大 skill 会线性撑爆每轮 prompt。超限按 personal>course>builtin 从尾部裁 + 省略行
# （与 render_skills_manifest 既有裁剪语义一致，不发明新规则）。
_ALWAYS_MAX_CHARS = 8_000

# builtin skill 随包发布（core/skills/builtin/），只读
BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent / "builtin"

# course 层根目录：data/skills/（每个 course_id 一个子目录，空 course → _global）
_USER_SKILLS_BASE = Path(BASE_DIR) / "data" / "skills"
# personal 层根目录：data/skills_user/（每个 user_id 一个子目录）
_PERSONAL_SKILLS_BASE = Path(BASE_DIR) / "data" / "skills_user"


@dataclass(slots=True)
class SkillInfo:
    name: str
    description: str
    source: str = "course"  # "personal" | "course" | "builtin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "read_only": self.source == "builtin",
        }


@dataclass(slots=True)
class SkillSummaryEntry:
    """manifest 一行：模型读取 skill 前看到的信息。"""

    name: str
    description: str
    available: bool = True
    missing: list[str] = field(default_factory=list)
    always: bool = False
    source: str = "course"  # "personal" | "course" | "builtin"（manifest 不用，list API 用）


class SkillNotFoundError(Exception):
    pass


class InvalidSkillNameError(Exception):
    pass


class InvalidSkillPathError(Exception):
    """read_skill_file 被要求读取 skill 目录外的路径时抛出。"""

    pass


class SkillExistsError(Exception):
    pass


class SkillReadOnlyError(Exception):
    """写入目标为 builtin（只读）skill 时抛出。"""

    pass


class InvalidSkillAlwaysError(Exception):
    """尝试给 course 层 skill 开启 always:true 时抛出（M-48）。"""

    pass


class SkillService:
    """SKILL.md 包的读取 + 运行时加载（personal + course + builtin 三层）。"""

    def __init__(
        self,
        *,
        user_root: Path,
        builtin_root: Path | None = BUILTIN_SKILLS_ROOT,
        personal_root: Path | None = None,
        is_shared_course_layer: bool = False,
    ) -> None:
        self._root = user_root  # course 层
        self._builtin_root = builtin_root
        self._personal_root = personal_root  # personal 层（None → 无个人层，教师场景）
        # 写入层：有 personal → 写 personal；否则写 course（现状）
        self._writable_root = personal_root if personal_root is not None else user_root
        # M-48：是否为"课程共享层"——只有经 get_skill_service(course_id, user_id="")
        # 构造的教师共享场景才为 True。直构的 SkillService（学生个人 / 测试通用层）
        # 默认 False，仍可写 always。always 在共享层会污染所有学生每轮 prompt，故禁止。
        self._is_shared_course_layer = is_shared_course_layer

    @property
    def root(self) -> Path:
        return self._root

    # ── path helpers ────────────────────────────────────────────────────

    def _validate_name(self, name: str) -> str:
        candidate = (name or "").strip().lower()
        if not _NAME_RE.match(candidate):
            raise InvalidSkillNameError("Skill name must match ^[a-z0-9][a-z0-9-]{0,63}$")
        return candidate

    def _resolve_skill_dir(self, name: str) -> tuple[Path, str] | None:
        """跨层定位 skill：personal > course > builtin。"""
        slug = self._validate_name(name)
        if self._personal_root is not None:
            personal_dir = self._personal_root / slug
            if (personal_dir / "SKILL.md").exists():
                return personal_dir, "personal"
        user_dir = self._root / slug
        if (user_dir / "SKILL.md").exists():
            return user_dir, "course"
        if self._builtin_root is not None:
            builtin_dir = self._builtin_root / slug
            if (builtin_dir / "SKILL.md").exists():
                return builtin_dir, "builtin"
        return None

    # ── parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return {}, content
        raw = match.group(1)
        body = content[match.end():]
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        return data, body

    def _load_info(self, skill_dir: Path, source: str) -> SkillInfo | None:
        file = skill_dir / "SKILL.md"
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            return None
        meta, _ = self._parse_frontmatter(text)
        return SkillInfo(
            name=skill_dir.name,
            description=str(meta.get("description") or "").strip(),
            source=source,
        )

    # ── public read API ─────────────────────────────────────────────────

    def list_skills(self) -> list[SkillInfo]:
        """所有可见 skill：personal + course + 未被覆盖的 builtin（按优先级去重）。"""
        out: list[SkillInfo] = []
        seen: set[str] = set()

        def _scan(root: Path | None, source: str) -> None:
            if root is None or not root.exists():
                return
            for entry in sorted(root.iterdir()):
                if not entry.is_dir() or not (entry / "SKILL.md").exists():
                    continue
                if not _NAME_RE.match(entry.name) or entry.name in seen:
                    continue
                info = self._load_info(entry, source)
                if info is not None:
                    out.append(info)
                    seen.add(info.name)

        _scan(self._personal_root, "personal")
        _scan(self._root, "course")
        _scan(self._builtin_root, "builtin")
        return out

    def read_skill_file(self, name: str, rel_path: str = "SKILL.md") -> str:
        """读取 skill 包内文件（read_skill 工具用）。

        rel_path 严格限制在 skill 目录内——绝对路径和穿越段（``..``）被拒绝。
        超过读取上限的内容带标记截断。
        """
        resolved = self._resolve_skill_dir(name)
        if resolved is None:
            raise SkillNotFoundError(name)
        skill_dir, _source = resolved

        candidate = (rel_path or "SKILL.md").strip() or "SKILL.md"
        rel = Path(candidate)
        if rel.is_absolute() or ".." in rel.parts:
            raise InvalidSkillPathError(f"Illegal skill file path: {rel_path}")
        target = (skill_dir / rel).resolve()
        if not target.is_relative_to(skill_dir.resolve()):
            raise InvalidSkillPathError(f"Illegal skill file path: {rel_path}")
        if not target.is_file():
            raise SkillNotFoundError(f"{name}/{candidate}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS] + "\n\n[... truncated ...]"
        return text

    def get_detail(self, name: str) -> dict[str, Any]:
        """skill 详情（name/description/always/content 正文/source），供管理 UI。"""
        resolved = self._resolve_skill_dir(name)
        if resolved is None:
            raise SkillNotFoundError(name)
        skill_dir, source = resolved
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        meta, body = self._parse_frontmatter(text)
        return {
            "name": skill_dir.name,
            "description": str(meta.get("description") or "").strip(),
            "always": bool(meta.get("always")),
            "content": body,
            "source": source,
            "read_only": source == "builtin",
        }

    # ── runtime loading (manifest + always) ─────────────────────────────

    def summary_entries(self) -> list[SkillSummaryEntry]:
        """每个可见 skill 的 manifest 行。"""
        entries: list[SkillSummaryEntry] = []
        for info in self.list_skills():
            resolved = self._resolve_skill_dir(info.name)
            if resolved is None:
                continue
            skill_dir, source = resolved
            meta, _ = self._parse_frontmatter(
                (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            )
            # M-48 纵深防御：course 层 skill 的 always 永不生效。
            # 即便有人绕过 API 直接在 course 目录写 always:true 文件，急切注入也会忽略它，
            # 杜绝课程作者污染所有学生每轮 prompt 的注入面。always 只认 personal/builtin。
            effective_always = bool(meta.get("always")) and source != "course"
            entries.append(
                SkillSummaryEntry(
                    name=info.name,
                    description=info.description,
                    available=True,  # 纯文本 playbook，无 requires gate
                    missing=[],
                    always=effective_always,
                    source=source,
                )
            )
        return entries

    def load_for_context(self, names: list[str]) -> str:
        """把给定 skill 的正文渲染成 system-prompt 块（仅 always:true 用）。

        预算控制（_ALWAYS_MAX_CHARS）：每个 always skill 原先裸 read_text 拼接无上限，多个大
        skill 会线性撑爆每轮 prompt。现按 personal>course>builtin 排序后累计，超预算从尾部
        （低优先级）pop + 追加省略行——与 render_skills_manifest 既有裁剪语义一致。
        """
        if not names:
            return ""
        # source 优先级：personal > course > builtin（靠前的更可能命中任务，保留更多）
        _src_rank = {"personal": 0, "course": 1, "builtin": 2}
        blocks: list[str] = []
        for name in names:
            resolved = self._resolve_skill_dir(name)
            if resolved is None:
                continue
            skill_dir, source = resolved
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            _, body = self._parse_frontmatter(text)
            body = body.strip()
            if not body:
                continue
            blocks.append((_src_rank.get(source, 9), f"### Skill: {name}\n\n{body}"))
        if not blocks:
            return ""
        blocks.sort(key=lambda x: x[0])  # 高优先级在前
        parts = [b for _, b in blocks]
        header = "## 常驻技能\n以下守则每轮都适用，优先于通用默认。\n\n"
        sep = "\n\n---\n\n"
        dropped = 0
        # 超预算从尾部（低优先级）pop；至少保留 1 个（最高优先级），即使它本身超预算
        while len(header + sep.join(parts)) > _ALWAYS_MAX_CHARS and len(parts) > 1:
            parts.pop()
            dropped += 1
        out = header + sep.join(parts)
        if dropped:
            out += f"\n\n（已省略 {dropped} 个低优先级常驻技能以控制 prompt 长度）"
        return out

    def load_always_for_context(self) -> str:
        """急切渲染 always:true 的 skill。"""
        names = [e.name for e in self.summary_entries() if e.always and e.available]
        return self.load_for_context(names)

    # ── public write API（写入层 = personal if exists else course）──────

    def _skill_dir(self, name: str) -> Path:
        return self._writable_root / self._validate_name(name)

    @property
    def _writable_is_course_layer(self) -> bool:
        """当前写入层是否为"课程共享层"（教师为课程制定、同课共享）。

        只有经 get_skill_service(course_id, user_id="") 构造的教师共享场景为 True
        （personal_root is None 且 is_shared_course_layer=True）。学生个人层、
        直构的通用 SkillService 均为 False，仍允许 always（那是用户自己的全局守则）。
        """
        return self._is_shared_course_layer and self._personal_root is None

    def _skill_file(self, name: str) -> Path:
        return self._skill_dir(name) / "SKILL.md"

    def _assert_writable(self, slug: str) -> None:
        """update/delete 前确认写入层存在目标；builtin 只读不可改。

        create 不走这里（create 用 target_dir.exists() 判重，允许在写入层建同名
        覆盖 builtin 的读取可见性）。
        """
        if (self._writable_root / slug / "SKILL.md").exists():
            return
        if self._builtin_root is not None and (self._builtin_root / slug / "SKILL.md").exists():
            raise SkillReadOnlyError(f"Skill is builtin (read-only): {slug}")
        raise SkillNotFoundError(slug)

    def create(
        self, name: str, description: str, content: str, always: bool = False
    ) -> SkillInfo:
        slug = self._validate_name(name)
        target_dir = self._skill_dir(slug)
        if target_dir.exists():
            raise SkillExistsError(slug)
        # M-48：course 层（教师为课程制定的共享 skill）不允许 always:true。
        # always 会让正文每轮急切注入所有学生的 system prompt；恶意/失误的课程作者可借
        # 此把任意指令塞进每轮对话。限制 always 只能来自 personal（学生自己的全局守则）
        # 或 builtin（随包发布、可信）。course 层 skill 仍可按需 read_skill 读取。
        if always and self._writable_is_course_layer:
            raise InvalidSkillAlwaysError(
                "课程共享 skill 不支持 always:true（会污染所有学生每轮对话）；"
                "请用个人 skill 或保留为按需读取"
            )
        body = self._normalize_content(slug, description, content, always=always)
        target_dir.mkdir(parents=True, exist_ok=False)
        self._skill_file(slug).write_text(body, encoding="utf-8")
        source = "personal" if self._personal_root is not None else "course"
        return SkillInfo(name=slug, description=description.strip(), source=source)

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        content: str | None = None,
        always: bool | None = None,
        rename_to: str | None = None,
    ) -> SkillInfo:
        slug = self._validate_name(name)
        self._assert_writable(slug)
        target_dir = self._skill_dir(slug)
        text = content if content is not None else self._skill_file(slug).read_text(encoding="utf-8")
        if description is not None:
            text = self._rewrite_frontmatter(text, description=description.strip())
        if always is not None:
            # M-48：course 层禁止开启 always（同 create 的理由）。
            if always and self._writable_is_course_layer:
                raise InvalidSkillAlwaysError(
                    "课程共享 skill 不支持 always:true（会污染所有学生每轮对话）"
                )
            text = self._rewrite_frontmatter(text, always=always)
        if rename_to and rename_to != slug:
            new_slug = self._validate_name(rename_to)
            new_dir = self._skill_dir(new_slug)
            if new_dir.exists():
                raise SkillExistsError(new_slug)
            text = self._rewrite_frontmatter(text, name=new_slug)
            target_dir.rename(new_dir)
            slug = new_slug
        self._skill_file(slug).write_text(text, encoding="utf-8")
        meta, _ = self._parse_frontmatter(self._skill_file(slug).read_text(encoding="utf-8"))
        source = "personal" if self._personal_root is not None else "course"
        return SkillInfo(
            name=slug,
            description=str(meta.get("description") or "").strip(),
            source=source,
        )

    def delete(self, name: str) -> None:
        slug = self._validate_name(name)
        self._assert_writable(slug)  # builtin 只读 → ReadOnly；不存在 → NotFound
        shutil.rmtree(self._writable_root / slug)

    # ── content helpers ────────────────────────────────────────────────

    def _normalize_content(
        self, name: str, description: str, content: str, *, always: bool = False
    ) -> str:
        """确保保存的文件有合法 frontmatter（name/description，可选 always）。"""
        text = content if content is not None else ""
        if _FRONTMATTER_RE.match(text):
            return self._rewrite_frontmatter(
                text, name=name, description=description.strip(), always=always
            )
        payload: dict[str, Any] = {"name": name, "description": description.strip()}
        if always:
            payload["always"] = True
        header = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()
        body = text.lstrip()
        return f"---\n{header}\n---\n\n{body}".rstrip() + "\n"

    def _rewrite_frontmatter(
        self,
        text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        always: bool | None = None,
    ) -> str:
        meta, body = self._parse_frontmatter(text)
        if name is not None:
            meta["name"] = name
        if description is not None:
            meta["description"] = description
        if always is not None:
            if always:
                meta["always"] = True
            else:
                meta.pop("always", None)
        if not meta:
            return text
        header = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{header}\n---\n\n{body.lstrip()}".rstrip() + "\n"


def render_skills_manifest(entries: list[SkillSummaryEntry]) -> str:
    """把 manifest 行渲染成 system-prompt 的 Skills 块。

    always 条目排除（其正文已急切注入，再列浪费 token）。重名保留首次出现。
    预算控制：单条 description 超长截断；总条数/总字符超限时按既有优先级从尾部裁剪
    （条目顺序已是 personal>course>builtin，靠前的更可能命中任务），并追加一行省略提示。
    """
    seen: set[str] = set()
    lines: list[str] = []
    for entry in entries:
        if entry.always or entry.name in seen:
            continue
        seen.add(entry.name)
        suffix = ""
        if not entry.available:
            suffix = f" (unavailable: {', '.join(entry.missing)})"
        description = entry.description or entry.name
        if len(description) > _MANIFEST_DESC_MAX_CHARS:
            description = description[: _MANIFEST_DESC_MAX_CHARS - 1].rstrip() + "…"
        lines.append(f"- **{entry.name}** — {description}{suffix}")

    if not lines:
        return ""

    header = (
        "## Skills\n"
        "按需可用的专项手册。当任务匹配某技能的描述时，先调用 `read_skill` "
        "读取其完整内容，再按指引执行。标注 unavailable 的技能在依赖满足前不可用。\n\n"
    )

    dropped = 0
    # 条数上限：超出按优先级从尾部裁剪
    if len(lines) > _MANIFEST_MAX_ENTRIES:
        dropped += len(lines) - _MANIFEST_MAX_ENTRIES
        lines = lines[:_MANIFEST_MAX_ENTRIES]
    # 字符预算：标题占固定空间，正文超限时继续从尾部移除（靠前的优先级更高，保留）
    budget = _MANIFEST_MAX_CHARS - len(header)
    while len("\n".join(lines)) > budget and len(lines) > 1:
        lines.pop()
        dropped += 1

    body = "\n".join(lines)
    if dropped:
        body += f"\n- …（另有 {dropped} 个技能因清单篇幅未列出）"
    return header + body


_instances: dict[str, SkillService] = {}


def get_skill_service(course_id: str = "", user_id: str = "") -> SkillService:
    """按 (course_id, user_id) 缓存的单例。

    - user_id 非空 → 含 personal 层（data/skills_user/<user_id>/），写入层 = personal
    - user_id 空  → 仅 course 层（data/skills/<course_id or _global>/），写入层 = course
    """
    cid = (course_id or "").strip() or "_global"
    uid = (user_id or "").strip()
    key = f"{cid}|{uid or '_'}"
    if key not in _instances:
        personal_root = _PERSONAL_SKILLS_BASE / uid if uid else None
        # uid 为空（教师/admin 场景）→ 写入层是课程共享目录，禁 always（M-48）；
        # uid 非空（学生）→ 写个人层，allow always。
        _instances[key] = SkillService(
            user_root=_USER_SKILLS_BASE / cid,
            personal_root=personal_root,
            is_shared_course_layer=not uid,
        )
    return _instances[key]


__all__ = [
    "BUILTIN_SKILLS_ROOT",
    "InvalidSkillAlwaysError",
    "InvalidSkillNameError",
    "InvalidSkillPathError",
    "SkillExistsError",
    "SkillInfo",
    "SkillNotFoundError",
    "SkillReadOnlyError",
    "SkillService",
    "SkillSummaryEntry",
    "get_skill_service",
    "render_skills_manifest",
]
