from __future__ import annotations
import asyncio, json, logging
from core.agent.tool_protocol import ToolResult
from core.observability.langsmith_trace import safe_traceable

logger = logging.getLogger(__name__)


# ── RAG Tool ────────────────────────────────────────────────────────────────

@safe_traceable(name="rag.retrieve", run_type="retriever")
async def _execute_rag(course_id: str, query: str, **kwargs) -> ToolResult:
    """调用 LightRAG 检索，返回 ToolResult。"""
    # 未选课 / 自由问答：无课程知识库，直接短路，避免对空库误检索
    if not course_id or course_id == "general":
        return ToolResult(content="（未选择课程，知识库不可用）", success=False)
    from core.rag import get_retriever
    try:
        retriever = get_retriever("lightrag")
        content = await retriever.retrieve_context(
            course_id=course_id,
            query=query,
            top_k=kwargs.get("top_k", 5),
            **kwargs,
        )
        if len(content) > 4000:
            content = content[:4000] + "\n...(truncated)"
        _preview = (content[:800] + "…") if len(content) > 800 else content
        logger.info(
            "tool_registry [rag] course=%s query_chars=%d retrieved_chars=%d empty=%s\n"
            "--- RAG 检索结果预览（前 800 字）---\n%s\n--- end preview ---",
            course_id,
            len(query),
            len(content),
            not bool(content),
            _preview or "（空）",
        )
        if content:
            logger.debug("tool_registry [rag] full retrieved_context:\n%s", content)
        return ToolResult(content=content, sources=[{"type": "rag", "query": query}])
    except Exception as e:
        logger.exception("rag tool failed")
        return ToolResult(content=f"（知识库检索失败：{e}）", success=False)


# ── WebSearch Tool ─────────────────────────────────────────────────────────

async def _load_user_search_override(user_id: str) -> dict | None:
    """读当前用户的搜索配置覆盖（UserSearchConfig）；无记录/空 user_id → None。

    async 层读好后传给同步 web_search（避免同步上下文碰 async DB）。
    """
    if not user_id:
        return None
    try:
        from core.db.database import AsyncSessionLocal, UserSearchConfig
        from sqlalchemy import select
        async with AsyncSessionLocal() as db:
            row = (
                await db.execute(
                    select(UserSearchConfig).where(UserSearchConfig.user_id == user_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "provider": row.provider or "",
                "api_key": row.api_key or "",
                "base_url": row.base_url or "",
                "max_results": row.max_results or 0,
                "proxy": row.proxy or "",
            }
    except Exception:
        logger.exception("load user search override failed user=%s", user_id)
        return None


async def _execute_web_search(query: str, *, user_id: str = "", **kwargs) -> ToolResult:
    """通过 services.search 多 provider 搜索，返回 ToolResult。

    user_id 非空时读该用户的搜索配置覆盖（UserSearchConfig），合并 admin 默认 + env。
    """
    try:
        from services.search import web_search

        # async 层读 user override（同步 web_search 不能查 async DB），再传给 web_search
        user_override = await _load_user_search_override(user_id)
        result = await asyncio.to_thread(web_search, query, user_override=user_override)

        # result 是 dict，含 answer / citations / search_results
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        search_results = result.get("search_results", [])

        # 优先用 answer（perplexity/tavily 等已合成）；
        # 无 answer 则降级到格式化 search_results（duckduckgo 等原始 SERP）
        if not answer:
            lines = []
            for i, r in enumerate(search_results[:5]):
                lines.append(f"[{i+1}] {r['title']}\n{r['url']}\n{r['snippet']}")
            answer = "\n\n".join(lines) or "（未找到相关网页）"

        # sources 供前端 SourceCard 展示
        sources = [
            {"type": "web", "title": c["title"], "url": c["url"]}
            for c in citations
        ] or [
            {"type": "web", "title": r["title"], "url": r["url"]}
            for r in search_results[:5]
        ]

        return ToolResult(content=answer, sources=sources)

    except Exception as e:
        logger.exception("web_search tool failed")
        return ToolResult(content=f"（网络搜索失败：{e}）", success=False)


# ── Ask User Tool ───────────────────────────────────────────────────────────

async def _execute_ask_user(questions: list, intro: str = "", **kwargs) -> ToolResult:
    """ask_user 工具：构造 pause_for_user payload，触发 loop 暂停等待用户回复。

    questions 格式：[{"id": "q1", "prompt": "你想要...？", "options": ["A", "B"]}]
    """
    payload: dict = {
        "intro": intro,
        "questions": questions if isinstance(questions, list) else [],
    }
    return ToolResult(
        content="[等待用户回复]",
        pause_for_user=payload,
    )


# ── Skill Tool（read_skill：渐进式揭示知识包）──────────────────────────────

async def _execute_read_skill(course_id: str, name: str, file: str = "SKILL.md", *, user_id: str = "", **kwargs) -> ToolResult:
    """read_skill 工具：读取 SKILL.md 知识包内容（模型按需拉取，避免 prompt 膨胀）。

    user_id 非空时跨 personal 层解析（学生私人 skill 覆盖课程 skill）。
    """
    from core.skills.skill_service import (
        InvalidSkillPathError,
        SkillNotFoundError,
        get_skill_service,
    )
    skill_name = str(name or "").strip()
    if not skill_name:
        return ToolResult(content="read_skill 需要 name 参数（技能名）。", success=False)
    rel = str(file or "SKILL.md").strip() or "SKILL.md"
    try:
        svc = get_skill_service(course_id, user_id)
        content = svc.read_skill_file(skill_name, rel)
        return ToolResult(
            content=content,
            sources=[{"type": "skill", "name": skill_name, "file": rel}],
        )
    except SkillNotFoundError:
        return ToolResult(content=f"（未找到技能：{skill_name}）", success=False)
    except InvalidSkillPathError:
        return ToolResult(content=f"（非法技能文件路径：{rel}）", success=False)
    except Exception as e:
        return ToolResult(content=f"（读取技能失败：{e}）", success=False)


async def _execute_load_tools(names, **kwargs) -> ToolResult:
    """load_tools 工具：加载 deferred 扩展工具 schema（渐进式工具发现）。

    loader 经 contextvar 注入（DynamicToolResolver 每 turn set）；LLM 只提供 names。
    """
    from core.agentic.dynamic_tools import current_deferred_loader
    loader = current_deferred_loader()
    if loader is None:
        return ToolResult(content="load_tools 在当前会话不可用（未配置扩展工具）。", success=False)
    name_list = [str(n).strip() for n in names] if isinstance(names, list) else []
    outcome = loader.load(name_list)
    parts: list[str] = []
    if outcome["loaded"]:
        parts.append("已加载（现在可调用）：" + "、".join(outcome["loaded"]))
    if outcome["already_loaded"]:
        parts.append("已加载过：" + "、".join(outcome["already_loaded"]))
    if outcome["unknown"]:
        parts.append("未知工具：" + "、".join(outcome["unknown"]) + "（请核对扩展工具清单中的准确名称）")
    return ToolResult(
        content="\n".join(parts) or "（无可加载工具）",
        success=not outcome["unknown"] or bool(outcome["loaded"]),
    )


# ── Solve Tools（确定性脊柱：plan / finish_step / replan）──────────────────
# capabilities/solve/tools.py。仅在 solve turn 激活（contextvar 有 session_id）。
# 智能在 loop 出口（模型规划+求解），这三个工具提供 commit plan、不跳步、bounded replan 的确定性。

def _no_solve_session() -> ToolResult:
    return ToolResult(content="当前无解题会话，solve 工具不可用。", success=False)


async def _execute_solve_plan(analysis: str, steps, **kwargs) -> ToolResult:
    """commit 解题计划。solve turn 第一步必须调用。"""
    from core.solve.session import current_solve_session_id, get_session, parse_steps
    sid = current_solve_session_id()
    if not sid:
        return _no_solve_session()
    parsed = parse_steps(steps)
    if not parsed:
        return ToolResult(content="solve_plan 需要非空 steps 数组，每项含 goal。", success=False)
    analysis = str(analysis or "").strip()
    session = get_session(sid)
    session.set_plan(analysis, parsed)
    first = session.next_step()
    payload = {
        "status": "planned",
        "analysis": analysis,
        "steps": session.map(),
        "next": first.to_dict() if first else None,
        "instruction": "现在用可用工具执行第一步，完成后调 solve_finish_step 传简短摘要。不要跳步。",
    }
    return ToolResult(content=json.dumps(payload, ensure_ascii=False), success=True)


async def _execute_solve_finish_step(step_id: str, summary: str, **kwargs) -> ToolResult:
    """标记当前步骤完成并前进；折叠该步的中间工具输出为摘要。"""
    from core.solve.session import current_solve_session_id, get_session
    sid = current_solve_session_id()
    if not sid:
        return _no_solve_session()
    step_id = str(step_id or "").strip()
    summary = str(summary or "").strip()
    session = get_session(sid)
    if not session.steps:
        return ToolResult(content="还没有计划。请先调 solve_plan 再调 solve_finish_step。", success=False)
    step = session.mark_done(step_id, summary)
    if step is None:
        return ToolResult(
            content=f"未知步骤 {step_id!r}；有效 id：{[s.id for s in session.steps]}。",
            success=False,
        )
    nxt = session.next_step()
    payload = {
        "status": "step_done",
        "completed": step_id,
        "next": nxt.to_dict() if nxt else None,
        "all_done": session.all_done(),
        "instruction": "现在写出最终答案。" if nxt is None else "执行下一步，完成后再次调 solve_finish_step。",
    }
    return ToolResult(content=json.dumps(payload, ensure_ascii=False), success=True)


async def _execute_solve_replan(reason: str, steps, **kwargs) -> ToolResult:
    """当前思路卡住时替换计划。有次数上限。"""
    from core.solve.session import current_solve_session_id, get_session, parse_steps
    sid = current_solve_session_id()
    if not sid:
        return _no_solve_session()
    parsed = parse_steps(steps)
    if not parsed:
        return ToolResult(content="solve_replan 需要非空 steps 数组，每项含 goal。", success=False)
    reason = str(reason or "").strip()
    session = get_session(sid)
    if not session.replan(reason, parsed):
        return ToolResult(
            content=json.dumps(
                {"status": "budget_exhausted", "instruction": "replan 预算已用尽，不要再 replan，基于已有材料收尾。"},
                ensure_ascii=False,
            ),
            success=False,
        )
    first = session.next_step()
    payload = {
        "status": "replanned",
        "reason": reason,
        "replans_used": session.replans,
        "replans_max": session.max_replans,
        "steps": session.map(),
        "next": first.to_dict() if first else None,
    }
    return ToolResult(content=json.dumps(payload, ensure_ascii=False), success=True)



# ── Registry ───────────────────────────────────────────────────────────────

TOOLS_OPENAI_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag",
            "description": "从课程知识库中检索与问题最相关的内容片段，适合回答课程知识点问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词，提炼自用户问题的核心关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取最新信息，适合知识库里没有的最新动态、时事或补充资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "向用户提出 1-4 个澄清问题（以卡片形式展示），暂停当前推理，"
                "等待用户回复后继续。仅在信息严重不足、无法推进时使用，"
                "不要滥用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "问题列表，最多 4 项",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "问题唯一标识（英文 snake_case）"},
                                "prompt": {"type": "string", "description": "向用户展示的问题文本"},
                                "options": {
                                    "type": "array",
                                    "description": "可选的预设选项列表（可省略，省略则自由输入）",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "prompt"],
                        },
                    },
                    "intro": {
                        "type": "string",
                        "description": "卡片顶部的引导语（可省略）",
                    },
                },
                "required": ["questions"],
            },
        },
    },
    # ── solve 确定性脊柱工具（仅 deep_solve turn 激活）─────────────────────
    {
        "type": "function",
        "function": {
            "name": "solve_plan",
            "description": (
                "制定解题计划：简短分析 + 有序步骤列表。必须最先调用。然后逐步用工具求解，"
                "每步完成后调 solve_finish_step。计划保持精简（2-6 步），简单题 1 步即可。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "description": "一两句话：题目要求什么、你的解法思路。"},
                    "steps": {
                        "type": "array",
                        "description": "有序步骤，每项 {goal}。goal 是简短的祈使句。",
                        "items": {
                            "type": "object",
                            "properties": {"goal": {"type": "string"}},
                            "required": ["goal"],
                        },
                    },
                },
                "required": ["analysis", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_finish_step",
            "description": (
                "标记当前步骤完成并前进。传入该步结论的简短摘要（关键结果/数值/结论）——"
                "它作为该步记录保留，中间工具输出被折叠。返回下一步，或表示全部完成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "description": "solve_plan 返回的步骤 id（如 'S1'）。"},
                    "summary": {"type": "string", "description": "该步结果的简短摘要，作为其记录。"},
                },
                "required": ["step_id", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_replan",
            "description": (
                "当前思路卡住或被证伪时替换计划。给出原因 + 新的有序步骤列表。"
                "有次数上限，仅用于真正的方向修正。预算耗尽则基于已有材料收尾。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "当前计划为何失败、要怎么改。"},
                    "steps": {
                        "type": "array",
                        "description": "新的有序步骤，每项 {goal}。",
                        "items": {
                            "type": "object",
                            "properties": {"goal": {"type": "string"}},
                            "required": ["goal"],
                        },
                    },
                },
                "required": ["reason", "steps"],
            },
        },
    },
]

# read_skill 是动态 schema（不进 TOOLS_OPENAI_SCHEMA 静态列表），由 ChatPipeline /
# DynamicToolResolver 在 context.skills_manifest 非空时注入。ReadSkillTool。
READ_SKILL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "读取某个技能的完整手册（SKILL.md）或技能包内的参考文件。"
            "当任务命中 Skills 清单中某个技能时，先读取该技能，再按其指引执行任务。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名，与清单中完全一致"},
                "file": {
                    "type": "string",
                    "description": "包内文件路径（如 references/api.md），默认 SKILL.md",
                },
            },
            "required": ["name"],
        },
    },
}


@safe_traceable(name="tool.execute", run_type="tool")
async def execute_tool(tool_name: str, course_id: str = "", user_id: str = "", **kwargs) -> ToolResult:
    """工具执行统一入口：全经 ToolRegistry（内置 + MCP 已统一注册）。

    形参刻意用 ``tool_name`` 而非 ``name``：工具名与业务参数命名空间必须隔离——
    read_skill 的业务参数也叫 ``name``（技能名），若本形参同名，调用
    ``execute_tool("read_skill", name="aihot")`` 会触发
    ``got multiple values for argument 'name'``。course_id 改默认值便于全关键字调用。
    """
    from core.agent.registry import get_tool_registry
    return await get_tool_registry().execute(tool_name, course_id=course_id, user_id=user_id, **kwargs)
