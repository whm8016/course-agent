"""SkillService — SKILL.md 知识包的三层存储与运行时加载（DeepTutor 式渐进式揭示）。

三层（读取优先级 personal > course > builtin）：
- personal: data/skills_user/<user_id>/  —— 用户私人 playbook（多租户隔离，学生侧）
- course  : data/skills/<course_id or _global>/ —— 教师为课程统一制定，同课共享
- builtin : core/skills/builtin/  —— 随包发布，只读

写入层：personal_root 存在 → 写 personal；否则写 course（教师场景保持原状）。
同名时 personal 覆盖 course 的可见性（学生可用私人版覆盖课程版）。

系统提示词只放每个 skill 的单行清单（``render_skills_manifest``），完整内容由
``read_skill`` 工具在任务匹配时拉取（避免 prompt 膨胀）。``always: true`` 的 skill
全文急切注入（用于每轮都适用的全局守则）。

忠实移植 DeepTutor ``services/skill/service.py``，裁掉 tag 词汇表 / hub 导入 /
requires·sandbox gate（教育场景纯文本 playbook 无外部依赖，available 永真）；
保留 user 层 CRUD（create/update/delete）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from config import BASE_DIR

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAX_READ_CHARS = 100_000

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


class SkillService:
    """SKILL.md 包的读取 + 运行时加载（personal + course + builtin 三层）。"""

    def __init__(
        self,
        *,
        user_root: Path,
        builtin_root: Path | None = BUILTIN_SKILLS_ROOT,
        personal_root: Path | None = None,
    ) -> None:
        self._root = user_root  # course 层
        self._builtin_root = builtin_root
        self._personal_root = personal_root  # personal 层（None → 无个人层，教师场景）
        # 写入层：有 personal → 写 personal；否则写 course（现状）
        self._writable_root = personal_root if personal_root is not None else user_root

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
            entries.append(
                SkillSummaryEntry(
                    name=info.name,
                    description=info.description,
                    available=True,  # 纯文本 playbook，无 requires gate
                    missing=[],
                    always=bool(meta.get("always")),
                    source=source,
                )
            )
        return entries

    def load_for_context(self, names: list[str]) -> str:
        """把给定 skill 的正文渲染成 system-prompt 块（仅 always:true 用）。"""
        if not names:
            return ""
        parts: list[str] = []
        for name in names:
            resolved = self._resolve_skill_dir(name)
            if resolved is None:
                continue
            skill_dir, _source = resolved
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            _, body = self._parse_frontmatter(text)
            body = body.strip()
            if not body:
                continue
            parts.append(f"### Skill: {name}\n\n{body}")
        if not parts:
            return ""
        return (
            "## 常驻技能\n以下守则每轮都适用，优先于通用默认。\n\n"
            + "\n\n---\n\n".join(parts)
        )

    def load_always_for_context(self) -> str:
        """急切渲染 always:true 的 skill。"""
        names = [e.name for e in self.summary_entries() if e.always and e.available]
        return self.load_for_context(names)

    # ── public write API（写入层 = personal if exists else course）──────

    def _skill_dir(self, name: str) -> Path:
        return self._writable_root / self._validate_name(name)

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
        lines.append(f"- **{entry.name}** — {description}{suffix}")
    if not lines:
        return ""
    return (
        "## Skills\n"
        "按需可用的专项手册。当任务匹配某技能的描述时，先调用 `read_skill` "
        "读取其完整内容，再按指引执行。标注 unavailable 的技能在依赖满足前不可用。\n\n"
        + "\n".join(lines)
    )


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
        _instances[key] = SkillService(
            user_root=_USER_SKILLS_BASE / cid,
            personal_root=personal_root,
        )
    return _instances[key]


__all__ = [
    "BUILTIN_SKILLS_ROOT",
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
