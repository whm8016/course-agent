# Backend Architecture

## Overview

The backend follows a **three-layer plugin model** : single-shot **Tools** (能做什么) + multi-stage **Capabilities** (怎么做) + on-demand **Skills** (应该知道什么 — knowledge packages the model fetches via `read_skill`). MCP server tools plug into the same `tool_calls` surface through a **progressive-disclosure** loader (`load_tools`). The four capabilities (chat / quiz / solve / research) all run on a shared `run_agent_loop`, driven by **tool_calls** (no label protocol layer).

```
Entry Points
 POST /api/chat (SSE streaming)
 WS /api/run/{cap} (WebSocket, unified)
 Bot QQ / Feishu (IM channels — shares the same engine)
 |
 TurnRuntimeManager + CourseOrchestrator (services/session/turn_runtime.py, core/orchestrator.py)
 |
 CapabilityRegistry (core/registry.py)
 ├── ChatCapability ──► ChatPipeline (system_prompt + run_agent_loop)
 ├── DeepSolveCapability ──► single loop + solve_plan/finish_step/replan tools + SolveSession
 ├── DeepResearchCapability──► rephrase → decompose → research(queue+parallel) → reporting(citation)
 └── QuizCapability ──► explore → plan → quiz
 |
 StreamBus (core/stream_bus.py)
 |
 SSE / WebSocket consumers
```

---

## Scheduling: Agent Loop

The core scheduling kernel lives in `core/agentic/loop.py` (`run_agent_loop`).

One user turn = one call to `run_agent_loop`:

```
for iteration in range(max_iterations):
 result = _one_round(messages, tool_schemas, model) # one streaming LLM call
 if result.has_tool_calls: # tool round
 emit thinking narration
 dispatch tools in parallel # core/agentic/tool_dispatch.py
 append role=tool messages
 continue
 else: # finish round (or last round tools=None)
 emit content as token events (true streaming, live_sink)
 break
```

The last iteration always runs with `tools=None`, forcing a text answer (forced finish + true chunk streaming; first-token latency ≈ model first token). Multi-stage capabilities (solve / question / research) are orchestrated in code as **multiple `run_agent_loop` calls** (one per stage), with `dataclasses.replace` to isolate context between stages.

### 通用上下文管理（in-loop context budget）

多轮工具调用会让 `messages` 滚雪球撑爆 context window（complexity trap）。重构为业界顶级实现通用的「绝对阈值 + 减法留白 + 分级级联 + 反应式兜底」架构，不绑定具体模型/供应商。调研依据：Claude Code 五级压缩级联、Anthropic Context Management API 参数语义、context rot 论文（arXiv:2606.29718）、When Refusals Fail（arXiv:2512.02445 标称窗口不可信）、arXiv:2508.21433（Observation Masking 成本减半）。

**窗口解析 + 双阈值**（`core/agentic/context_window.py`）：`resolve_effective_window(model)` 四级解析（显式配置 `LLM__CONTEXT_WINDOW` / catalog -> 探测缓存 `data/context_window_cache.json` -> 模型名模式 `_MODEL_WINDOWS` -> heuristic `max(16384, max_tokens×4)` 兜底并 `log_flow` 告警，杜绝旧实现「未知模型静默按 32768 跑」）。探测层（`core/agentic/window_probe.py`，对标 DeepTutor `detect_context_window`）在请求热路径之外跑--`main.py` lifespan 启动预热 + admin `POST /admin/context-window/probe` 手动重探，`GET {base_url}/models` 递归扫 8 个候选字段 + model id 别名匹配拿真实窗口，落缓存（键=base_url+model，带 detected_at + 7 天 TTL）；热路径 `resolve_effective_window` 只同步读缓存、零网络开销，探测拿不到时退回 `_MODEL_WINDOWS` 表（原生 OpenAI /models 不暴露窗口属正常降级）。`resolve_effective_window_with_source` 返回 `(window, source)` 供 admin 端点展示当前生效层级（explicit/probe/table/heuristic）。`compute_budgets(model)` 返回 `(soft_trigger, hard_ceiling)`，减法留白：`hard_ceiling = window - min(max_output, output_reserve) - safety_margin`，`soft_trigger = min(rot_threshold_tokens, int(window * quality_ratio), hard_ceiling)`（三取 min：窗口比例线 quality_ratio，RULER 实测有效长度多为窗口 0.25-0.5 取 0.5；绝对上限 rot_threshold_tokens 在比例线算出过大时钳制，rot 论文 64k 为成本可接受上界；再被硬天花板钳制。128k 模型算出 64000 与旧默认逐字相同，1M 模型放开到 128k）。软阈值驱动主动压缩追求质量，硬天花板驱动紧急降级保证可用。

**三级级联执行器**（`core/agentic/context_budget.enforce`，loop 每轮调一次，永远先用最便宜的手段）：L1 清旧 tool 结果（免费，`keep_recent_turns` + `exclude_tools` 白名单保护 ask_user 不可重建输入）-> L2 LLM 摘要续接（L1 后仍超软阈值才触发，`<context_summary>` 前插）-> L3 丢最旧 20% 消息组（仍超硬天花板，保请求发得出去）。未超软阈值时 L1 noop（等价 raw）。取代旧「窗口百分比 + 策略互斥开关」四路分支。

**反应式回合前裁剪**（`plan_turn`）：未超软阈值直接原样返回（零信息损失），超了才裁历史 + 摘要续接（`carry_forward_location="history_prefix"` 时前插 `<session_summary>`，turn_runtime 据此清 system 切片避免双付）。`coordinator_enabled` 开启时由 `turn_runtime.start_turn` 调用（默认关=旧 `resolve_budget` 20% 裁历史）。

**反应式兜底 + 熔断**（`loop._one_round`）：估算失误（tokenizer 差异/多模态/工具结果突增）导致 provider 拒 context-length 时，紧急 L3 丢最旧 20% 后重试；连续失败 `max_consecutive_failures`（默认 3）次熔断，返回错误而非无限重试烧钱。对标 Claude Code Reactive Compact + `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES=3`。

**评测臂**（`context_policy.apply_arm`，contextvar `set_arm` 覆盖，对照 arXiv:2508.21433）：`raw`/`masking`/`summary_only`/`hybrid` 四臂经 contextvar 注入（串行 set->跑->reset 不污染下一组合，生产零侵入）。掩码/摘要改 token 口径（与级联同一 `soft_trigger`）；`cap_tool_result` 单条 6000 字段落边界截断仍在 tool_dispatch 入口即裁（RECOMP arXiv:2310.04408 思路）。消融 runner 见 `scripts/eval_context/`。

**Skill 预算**（`core/skills/skill_service.load_for_context`）：always-on skill 全文合计超 `_ALWAYS_MAX_CHARS` 时从尾部裁低优先级（personal 优先保留），防 skill 清单本身吃掉上下文。

### tool_calls vs label-driven (the )

| Approach | How "what next?" is decided |
|---|---|
| ReAct (label protocol) | First-line label (`FINISH`/`TOOL`/`THINK`) — |
| **tool_calls (this project)** | **OpenAI `tool_calls` field** — provider-guaranteed structure, works across GPT/Qwen/DeepSeek |

This project chose tool_calls: simpler prompts, provider-guaranteed formatting, mainstream. Where relies on the label protocol to enforce flow gating, this project achieves equivalence via **code orchestration + tool state machines** (SolveSession for solve, three-stage orchestration for quiz, DynamicTopicQueue + CitationManager for research).

---

## Request Flow: POST /api/chat

```
POST /api/chat
 └── api/chat.py
 ├── auth + check_course_access (multi-tenant isolation)
 ├── build UnifiedContext
 ├── TurnRuntimeManager.start_turn(ctx)
 └── SSE: subscribe_turn(turn_id) → yield events
 └── (TRM drives) CourseOrchestrator → ChatCapability → ChatPipeline
 ├── KB Seed 预检索（retrieve_kb_seed：进 loop 前用原问题查一次课程知识库，命中则作 [知识库预检索] 块前置注入；未命中/超时/未挂 rag → 空串降级，loop 行为零变化）
 ├── assemble system_prompt (chat.yaml loop spec + course prompt + persona + memory)
 └── run_agent_loop(extra_context=kb_seed, temperature=0.3)
```

Key files:

| File | Role |
|------|------|
| `api/chat.py` | SSE endpoint, TRM lifecycle |
| `services/session/turn_runtime.py` | TurnRuntimeManager — turn lifecycle + StreamBus fan-out |
| `core/orchestrator.py` | `CourseOrchestrator` — routes context to the selected capability |
| `core/agent/mode_normalize.py` | `normalize_mode`（chat_mode 规范化）|
| `core/capabilities/chat_pipeline.py` | system_prompt assembly + run_agent_loop |
| `core/agentic/loop.py` | `run_agent_loop` — the scheduling kernel (tool_calls, true streaming) |
| `core/agentic/tool_dispatch.py` | parallel tool execution (≤8 concurrent) |
| `core/agent/tool_registry.py` | rag / web_search / ask_user / solve_* + read_skill / load_tools / mcp_* dispatch |
| `core/agentic/dynamic_tools.py` | `DynamicToolResolver` — assemble tool_schemas (base + read_skill + load_tools + deferred mcp_*), contextvar-injected loader |
| `core/skills/skill_service.py` | `SkillService` — SKILL.md packages (builtin + per-course user layers, manifest + read_skill_file) |
| `core/mcp/` | MCP connection manager (config 部署级 + manager 单例 lifecycle + adapter + deferred_tools + session_state) |
| `core/prompt_loader.py` | YAML prompt loader (shared by all four capabilities) |
| `core/stream_bus.py` | async event bus with replay |

---

## The Four Capabilities 

| Capability | Pipeline | Stages (tool_calls version, code-orchestrated) |
|---|---|---|
| chat | `core/capabilities/chat_pipeline.py` | run_agent_loop (single loop) |
| deep_solve | `core/solve/pipeline.py` | single loop + solve_plan/finish_step/replan tools + `SolveSession` (session_id injected via contextvar) |
| quiz | `core/question/pipeline.py` | explore → plan (JSON templates) → quiz (per-question JSON + schema validation, 6 question types; Stage 3 parallelizes the N independent questions via `asyncio.gather` + `Semaphore` — `_fork_for_quiz` deep-copies attachments/metadata per question so concurrent `_build_messages` calls don't clobber each other) |
| deep_research | `core/research/pipeline.py` | rephrase → decompose → research (`DynamicTopicQueue` + parallel) → reporting (outline/intro/sections/conclusion + `CitationManager`) |

Each capability externalizes its prompts to `core/<cap>/prompts/zh/*.yaml`, loaded via `load_prompt_dict`.

### 深度研究：前置澄清 + 可恢复暂停（plan: research_clarification_pause_resume）

- **前置澄清**：rephrase 阶段挂 `ask_user`（双重 gate：`settings.research.clarify_enabled` 且 WS 入口注入了 `wait_for_user_reply`），主题模糊时在同 turn 内暂停问学生、回复后续跑；HTTP/IM 无 waiter 不挂（否则 loop 无 waiter 直接结束）。`_REPHRASE_TOOLS` 与 `_RESEARCH_TOOLS` 分开——研究块/报告不问用户（中后期提问收益为负）。waiter 硬超时 `clarify_wait_timeout_s`（默认 120），超时返回 skip payload（`_format_reply`→"User skipped."续跑），不返回 None（None 被 loop 当取消）。
- **跨 worker 回复**（`services/session/reply_channel.py`）：ask_user 回复走 Redis `BLPOP`/`RPUSH`（`ca:askuser:reply:{turn_id}`），任意 worker 的 `submit_user_reply` 都能投到正在等待的 worker，不再依赖 Nginx `ip_hash`；`memory://`/连不上回退进程内 `asyncio.Queue`（降级范式照 `core/arq_pool.py`）。turn 归属 key `ca:turn:owner:{turn_id}` 供跨 worker IDOR 校验（本地 `_executions` miss 时回落）。
- **阶段级 checkpoint**（`core/research/checkpoint.py` + 迁移 `022` + `ResearchCheckpoint` 模型）：`pipeline.run()` 每阶段末存 checkpoint（phase + state_json，含 `DynamicTopicQueue.to_dict`）；`resume_research_id` 命中→`_phases_done(phase)` 跳过已完成阶段、被中断阶段整段重放（接受该阶段内部 LLM/检索成本重付一次，与 LangGraph「node 从头重跑」同语义）。`research_id` 随 `stage_start "init"` 提前下发，前端保存供断线重连。
- **awaiting_user 恢复**：loop 暂停前调 `context.metadata["on_ask_user_pause"]` 回调（pipeline 注入）落 `status=awaiting_user`+卡片；重连时 pipeline 重发同一份卡片、等回复、把回复当 refined_topic。**注意：awaiting_user 判断必须在「phase∈done_phases」之前**——awaiting_user 行的 phase 常已是 rephrase，否则跳过分支先命中、永远不恢复卡片。

---

## SSE Event Protocol

| Event type | When emitted |
|---|---|
| `thinking` | LLM narration before a tool call |
| `tool_call` / `tool_result` | before / after a tool invocation |
| `token` | one chunk of the final answer (true streaming) |
| `answer` / `done` | full final answer / turn end (includes tools_used, mode, iterations) |
| `stage_start` / `stage_end` | stage boundaries (explore/plan/quiz/rephrase/...) |
| `quiz_question` / `result` | per-question in quiz / capability finalization |
| `quiz_question_error` | a single quiz question failed/invalid (M-8/M-9: per-question fault tolerance — other questions still generate) |
| `progress` | sub-stage progress; in research, parallel blocks emit `block_start`/`block_end` with `block_id` (M-10: per-block event isolation — each block's events flush as one contiguous segment, not interleaved) |
| `error` | unrecoverable failure |

---

## Tools

Defined in `core/agent/tool_registry.py` as OpenAI function schemas + async executors:

| Tool | Description |
|------|-------------|
| `rag` | LightRAG knowledge base retrieval |
| `web_search` | 多 provider web search（perplexity / tavily / duckduckgo 等，见 `services.search`） |
| `ask_user` | pause the loop to ask the user for clarification (WS entry only) |
| `solve_plan` / `solve_finish_step` / `solve_replan` | solve deterministic spine (deep_solve turn only; session_id injected via contextvar) |
| `read_skill` | dynamic schema — mounted when `context.skills_manifest` is non-empty; fetches a SKILL.md playbook |
| `load_tools` | dynamic schema — mounted when deferred MCP pool is non-empty; loads MCP tool schemas into the turn |
| `mcp_<server>_<tool>` | MCP server tools (deferred by default; registered into ToolRegistry, executed via `registry.execute`) |
| `cron` | IM bot 对话内设定时提醒（仅 bot 渠道挂载；owner 经 contextvar 注入，见 `core/bot/cron_tool.py`） |
| `query_timetable` / `query_grades` / `query_mistakes` | 只读学业查询（个人课表 / 成绩 / 错题本），见下方「学业结构化工具」 |

The built-in tools (`rag`/`web_search`/`ask_user`/`solve_*`) are **context-gated**: `context.enabled_tools` controls which are mounted. The dynamic tools (`read_skill` / `load_tools` / `mcp_*`) are assembled by `DynamicToolResolver` (see below), not the static `TOOLS_OPENAI_SCHEMA`.

### 学业结构化工具（`core/academic/tools.py`）

3 个**只读**学业查询工具：`query_timetable`（个人课表，JOIN Enrollment 跨已选课程）、`query_grades`（本人成绩，按注入的 `course_id` 收窄）、`query_mistakes`（错题本，读 `NotebookEntry` 中 `is_correct=False`）。与 `rag` 的语义边界由 `chat.yaml` 的「工具选择原则」全局兜底（问「我的」结构化信息走结构化工具，问「课程讲了什么」走知识库）。

权限设计（OWASP LLM06 / 读写分离 / 最小权限）：
- **身份只走注入**：`user_id` 由 `registry.execute` 注入（`registry.py:100`），schema **绝不**暴露 `user_id`/`student_id` 参数。若模型幻觉出该参数，`tool_dispatch.py` 的 `**call_kwargs` 会与注入的 `user_id` 撞成 `TypeError` → registry 兜底「工具执行失败」（fail-closed）。
- **参数化查询**（SQLAlchemy ORM）而非 text-to-SQL，身份来自注入值不来自模型输入，把越权在结构上变成不可能：`query_grades` 强制 `WHERE student_id == 注入值`。
- **全部只读 SELECT**；写入只走教师 REST API + JWT 角色校验（`api/teacher.py` 的 schedule/grades 端点 + `_get_owned_kb` owner 校验）。
- 空结果返回 `success=False` 明确无命中，不抛异常、不带 `sources`。

---

## Progressive Disclosure: Skills + Deferred MCP

Both **Skills** (knowledge playbooks) and **MCP tools** (deferred) reach the model the same way — the system prompt carries a **one-line manifest** per item, and the model fetches the full content on demand. This keeps the always-on schema surface small (improves tool selection on weaker models) while keeping every skill/tool one cheap call away. Two guards keep that surface from re-bloating under load: the manifest is **budget-capped** (`render_skills_manifest` truncates over-long descriptions and drops tail entries by `personal>course>builtin` priority once an entry/char ceiling is hit, appending an ellipsis line), and `read_skill` **dedupes within a turn** — re-reading the same `(name, file)` returns a pointer to the prior result instead of re-injecting the full text; the dedup set is a `chat_pipeline`-scoped `contextvars.ContextVar` (same lifecycle pattern as the deferred-tool loader).

| Mechanism | Manifest block | Fetch tool | What it reveals |
|---|---|---|---|
| **Skills** | `context.skills_manifest` (rendered by `SkillService`) | `read_skill(name, file?)` | full SKILL.md playbook (+ optional `references/`) |
| **Deferred MCP** | `context.extended_tools_manifest` (rendered by `render_deferred_tools_manifest`) | `load_tools(names)` | OpenAI tool schema → appended to the live `tool_schemas` list |

- **`SkillService`** (`core/skills/`) loads `SKILL.md` packages from two layers (builtin read-only + per-course user that shadows builtin). `always: true` skills are eagerly injected; the rest are manifest-only.
- **`MCPConnectionManager`** (`core/mcp/`) owns per-server connection tasks (each holds its own `AsyncExitStack` — MCP SDK cancel scopes are task-bound, so sessions must open/close within one task). MCP tools default to `deferred=True`: not in the initial tool list, reachable only via `load_tools`. Loaded names persist per chat session (`session_state`).
- **`DynamicToolResolver`** (`core/agentic/dynamic_tools.py`) assembles each turn's `tool_schemas` (base + read_skill + load_tools + already-loaded mcp_*), binds the `DeferredToolLoader` to that **mutable list**, and exposes the loader via a `contextvars.ContextVar`. **`run_agent_loop`'s core loop is unchanged** — `loop.py:252` reuses the same list reference each iteration (`schemas = None if final else tool_schemas`), so `load`'s `.append` makes the tool callable on the very next round.

---

## Capability Registry

`core/registry.py` → `CapabilityRegistry` maps names to instances:

```python
registry.register(ChatCapability) # "chat"
registry.register(DeepSolveCapability) # "deep_solve"
registry.register(DeepResearchCapability) # "deep_research"
registry.register(QuizCapability) # "quiz"
```

Adding a new capability: implement `BaseCapability` (see `core/capability_protocol.py`), call `registry.register`.

**`TrackedCapability`（`capability_protocol.py`）** — solve / quiz / research 三个能力的 `run()` 共享「计时 + 异常兜底」骨架（`log_flow <name>.error` + `logger.exception` + `stream.error(source=<name>)`），由 `TrackedCapability` 的模板方法 `run()` 统一兜底；子类只实现 `run_with_tracking`（跑流水线 + 自己打 start/complete 业务日志 + emit success 事件）与 `error_label`（中文文案前缀）。chat 是裸委托薄壳，直系 `BaseCapability`、不继承 `TrackedCapability`。

---

## Bot Sub-system

The IM Bot (`core/bot/`) **already shares the same Agent engine as the Web API**: `core/bot/agent/loop.py` routes through TurnRuntimeManager + CourseOrchestrator + ChatPipeline + `run_agent_loop` (no longer an independent thin shell), so it automatically inherits course prompts, DB memory updates, and LLM circuit-breaker/fallback. QQ/Feishu messages flow: MessageBus → AgentLoop → TRM.

---

## Key Design Decisions

1. **tool_calls driven (no label layer)**: provider-guaranteed structured output; flow gating done in code (SolveSession / three-stage orchestration / queue).
2. **StreamBus fan-out**: SSE, WebSocket, and tests all subscribe to the same bus — no duplication between transport layers.
3. **Capability isolation**: each capability owns its full pipeline; the orchestrator only routes, never executes business logic.
4. **Externalized prompts**: `core/<cap>/prompts/zh/*.yaml` + `prompt_loader`, 's prompt organization.
5. **Forced finish + true streaming**: the last loop iteration forces `tools=None` to guarantee a text answer, with chunk pass-through for low first-token latency.
6. **Progressive disclosure (Skills + deferred MCP)**: the system prompt carries one-line manifests; `read_skill` reveals knowledge playbooks, `load_tools` reveals tool schemas — isomorphic mechanisms. `run_agent_loop`'s core loop is **unchanged**: a mutable `tool_schemas` list + `contextvars`-injected loader makes deferred tools appear next round with zero edits to the scheduling kernel.

---

## Schema 设计原则

数据库 schema 的完整设计规范（"宪法"）见 **[`SCHEMA_DESIGN_PRINCIPLES.md`](./SCHEMA_DESIGN_PRINCIPLES.md)**——7 条原则 + 27 表体检打分 + P0–P4 收尾路线图。新表/迁移/重构以其为准绳。三条最常踩的核心原则固化如下：

1. **分层归属（一概念一真相源）**：每表归属原始事件 / 派生读模型 / 可变状态 / 引用配置 之一；一个概念只能有一个权威存储（如 KB 构建状态只在 `kb_builds`，不在 KB 行死列）。
2. **读写分离（CQRS）**：原始层只 INSERT、内容永不 UPDATE；rollup 层 cron 重算（delete-then-reinsert），读路径只读 rollup 不现算。
3. **多租户就地隔离**：凡课程级可查的行，**写时落盘 `course_id`**，使查询无需 JOIN Session 即可确定租主（`messages`/`notebook_entries`/`bot_notifications` 当前缺此列，见路线图 P1）。

其余四条（FK 故意分裂 / 可变列 OCC 或 append / Expand/Contract 演进 / ID-时间戳约定一致）详见宪法文档。

---

## Frontend Integration

Frontend (`frontend/`, React 19 + TypeScript + Vite + Tailwind) talks to the backend through two entry points; all four capabilities run on the new pipelines (old WS endpoints retired).

| Capability | Frontend entry | Backend route | Pipeline |
|---|---|---|---|
| chat | `chatStream` (SSE) | `POST /api/chat` (mode=chat) | ChatPipeline |
| deep_solve | `chatStream` (SSE, chatMode) | `POST /api/chat` (mode=deep_solve) | DeepSolvePipeline (solve tools) |
| quiz | `connectQuestionGenerate` (WS) | `WS /api/run/quiz` | QuizPipeline (explore→plan→quiz) |
| research | `connectDeepResearch` (WS) | `WS /api/run/deep_research` | ResearchPipeline (4 stages) |

`/api/chat` sets `mode = normalize_mode(chat_mode)` and routes via CourseOrchestrator (not hardcoded to ChatPipeline). `/api/run/{cap}` is the unified WS entry (`api/run.py`). Old endpoints (`/api/deep-research/run`, `/api/question/generate`) are retired — frontend migrated to the unified WS.

**Management APIs** (new, for admin/teacher UIs): `/api/skill-knowledge` (skill package CRUD, teacher), `/api/mcp/*` (MCP server config + `/probe`, admin). Existing `/api/skills` remains the **output-cards** (对话后补充框) — a different concept from skill knowledge packages; routes kept separate to avoid the naming clash.

**Three sides**: student (four capabilities + notebook + dashboard/graph), teacher (TeacherPage private: courses/KB/students/analytics — **untouched** + skill management), admin (AdminPage private: KB/users/invite-codes — **untouched** + MCP config). Shared UI components in `frontend/src/components/ui.tsx`; capability-specific rendering: `ThinkingProcess` shows `tool_call` steps (deep_solve solve_plan/finish_step) + `stage_start` progress (research).

---

## Multi-worker Deployment

生产部署使用 `gunicorn -w 4`（4 个 UvicornWorker）。多 worker 下进程隔离导致以下问题：

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| CronService ×4 | 定时任务执行 4 次 | Leader 选举，仅 leader 启动 Cron |
| TutorBotManager ×4 | IM 消息重复处理 | Leader 选举，仅 leader 启动 Bot |
| MCP 工具对非 leader worker 不可见 | 落在非 leader worker 的对话请求看不到任何 MCP 工具（曾错误套用 leader-only 模式，见下方说明） | **不**用 leader 选举：每个 worker 独立 `ensure_started()` + 周期 `reload()`（间隔 `settings.mcp.reload_interval_s`，`main.py` `_mcp_reload_loop`） |
| TurnRuntime KeyError | SSE 断线重连路由到不同 worker → `_executions` 找不到 | Nginx `ip_hash` 粘性会话 |
| Circuit Breaker 语义减弱 | 每个进程独立 failure_threshold → 总阈值放大 N 倍 | 参数缩放：`failure_threshold // workers`（最小 2） |
| LightRAG LRU 缓存 | 4 份缓存，内存占用 ×4 | 容量缩放：`capacity // workers` |

### MCP 为什么不用 leader 选举

Cron/Bot 的副作用发生在 DB/外部渠道，"谁执行都一样"，其他 worker 不需要感知，适合 leader-only。
MCP 连接不同：它的 adapter 要被**每个 worker 各自处理的对话请求**用到，性质更接近 DB 连接池（每
个 worker 都要有一份能用的连接）。若套用 leader-only（早期版本如此），单容器 `gunicorn -w N` 下只
有 leader worker 建立了连接，其余 worker 的 `get_mcp_manager()` 是空的——落在它们身上的请求会
静默看不到任何 MCP 工具（`dynamic_tools.resolve()` 的 `pool` 为空，不报错，只是清单里没有）。且
nginx 侧只有单个 `server backend:8002`（见 `frontend/nginx.conf`），`ip_hash` 在只有一个上游节点
时不影响 gunicorn 内部 4 个 worker 间的请求分配，无法保证请求总落到 leader 上。

现改为：每个 worker 在 `lifespan()` 里各自 `ensure_started()`（`main.py`），并起一个周期
`_mcp_reload_loop` 调 `manager.reload()`——因为 admin 保存/删除 MCP server 配置（`POST`/`DELETE
/api/mcp/servers/{name}`）只会在受理该请求的那个 worker 里触发一次 `reload()`，其余 worker 靠这
个轮询跟上配置变化（`reload()` 内部按 `connection_signature()` 比对，未变化的 server 不重连，轮
询开销可忽略）。轮询间隔配置化 `settings.mcp.reload_interval_s`（默认 30s，`.env` `MCP__RELOAD_INTERVAL_S`）；
项目无 Redis pub/sub 基建，「改配置」低频，最终一致即可，不建跨进程消息通道。stdio N 份子进程的代价
已由下方 gateway sidecar 消除。

### MCP gateway sidecar（`mcp-gateway/` + `docker-compose.yml`）

`_mcp_reload_loop` 解决了「非受理 worker 感知配置变化」，但 stdio server 仍有「每 worker 各起一份
子进程 = 4N 份常驻」的内存代价。引入独立 `mcp-gateway` 容器收敛：容器内一份 `mcp-proxy`（sparfenyuk
版，transport 桥）把多个 stdio server 暴露成命名 SSE 端点（`/servers/{name}/sse`），backend 的 4 个
worker 全改以 HTTP 客户端接入这一个 gateway，stdio 子进程全网只剩 gateway 内那一份（4→1，省内存）：

```
backend（4 worker）──SSE──> mcp-gateway（1 份 mcp-proxy）
                                  ├─ modelscope-mcp-server（本地，替代不可靠的云端 SSE 接口）
                                  └─ mcp-server-fetch（替代每 worker 一份的 uvx 子进程）
deepwiki 直连官方 https://mcp.deepwiki.com/mcp（streamableHttp，免认证无子进程），不走 gateway
```

- **构建期预装、运行时零网络**：gateway Dockerfile 用 `uv tool install` 把 server 版本钉死装进独立
  venv，运行时直接 exec 已装入口，不再 `uvx` 运行时拉包；版本变更走一次 build → 可复现/可回滚/可审计
  （对冲 `@latest` 的 installer spoofing）。`UV_OFFLINE=1` 兜底，意外拉包快速失败而非卡超时。
- **mcp 版本隔离**：`modelscope-mcp-server`（mcp 2.x）与 `mcp-server-fetch`（mcp<2）各自独立 tool venv，
  互不冲突；`--pass-environment` 把 `MODELSCOPE_API_TOKEN` 透传给子进程。
- **transport 选 SSE**：sparfenyuk/mcp-proxy 服务端只暴露 SSE，manager 的 `sse_client` 成熟且有测试；
  backend 每个 server 配 `type=sse` + `url=http://mcp-gateway:8096/servers/{name}/sse`。
- **资源隔离**：gateway 独立容器、不挂 `/app` 业务卷、`mem_limit: 1g`，第三方代码读不到业务数据。
- **lazy 解耦**：backend 不 `depends_on` gateway——MCP 懒连接（首 turn），gateway 未就绪只令该 server
  标 `error`，不阻塞 backend 启动。

### Leader 选举（`core/leader.py`）

Redis SETNX 效率型锁（单实例 Redis 足够，无需 Redlock）+ **竞选者循环 + CAS 原子续约**，TTL 30s：

- **启动抢锁**：`SET worker:leader <pid-uuid> NX EX=30`（worker_id 全局唯一 —— pid 跨重启会复用）。
- **leader 续约**：每 15s 用 **Lua CAS** 原子续约（`if GET==self then EXPIRE`），消除 `GET+EXPIRE` 竞态（否则会误续别人抢到的锁 → 双 leader 脑裂、单例跑两份）。
- **非 leader 竞选**：每 10s 重试 SETNX（不再「启动抢一次就放弃」）。leader 卡死但未被进程管理器杀死时，锁 TTL 过期后秒级接管 —— 空窗 ≤ TTL + 竞选间隔 ≈ 40s，而非等 Gunicorn `--timeout 300` 杀进程的分钟级。
- **状态翻转回调**：成为 leader → `on_gain` 拉单例；丢锁（CAS 返回 0）→ `on_lose` 停单例 → 重回竞选，闭环。

```python
register_leader_callbacks(on_gain=start_singleton_services, on_lose=stop_singleton_services)
await try_become_leader # 当选走 on_gain；未当选起竞选 loop，接管时再走 on_gain
```

`is_leader` 动态反映当前状态（竞选接管 / 丢锁实时更新），不再是启动时的静态快照。状态上报 Prometheus `ca_leader_is_leader`（`sum==0` 即无 leader 告警）。

### 粘性会话（`frontend/nginx.conf`）

```nginx
upstream backend_pool {
 ip_hash; # 同一客户端 IP 路由到同一 worker
 server backend:8002;
}
```

TurnRuntime 的 `_executions` 字典在进程内，断线重连必须回到同一 worker。`ip_hash` 保证客户端 IP 粘性。

### 参数缩放

| 资源 | 单 worker 值 | 多 worker 公式 |
|------|-------------|---------------|
| Circuit Breaker failure_threshold | 5 | `5 // workers`（最小 2） |
| LightRAG LRU capacity | 128 | `128 // workers`（最小 2） |

**原理**：总资源量不变，但分散到各进程。缩放后语义等价（总并发上限相同），避免 4 worker × 25 = 100 并发过载。

---

## Observability: LangSmith Tracing

全链路追踪覆盖 LLM 调用、工具调用、RAG 检索结果。未启用时零开销 no-op。

### 集中模块（`core/observability/langsmith_trace.py`）

```python
_TRACING_ON = bool(LANGSMITH_TRACING) and bool(LANGSMITH_API_KEY)

def is_tracing_enabled -> bool: ...
def safe_traceable(*, name, run_type, ...): ... # 未启用时 identity 装饰器
async def trace_context(*, name, metadata, tags): ... # 顶层 root run
def wrap_openai_client(client, *, chat_name): ... # AsyncOpenAI/AsyncAzureOpenAI wrap
```

**设计原则**：
- **LLM 层靠 `wrap_openai`**：不重复用 `@traceable`，避免双重 run
- **非 LLM 层靠 `@safe_traceable`**：工具/RAG/LightRAG 内部 LLM
- **顶层 `trace_context`**：`_run_turn` 外层建立 root run，下游自动嵌套
- **降级第一公民**：LangSmith 异常被吞掉，绝不阻塞业务路径

### 检测点（7 处）

| 位置 | 装饰/包装 | 覆盖的盲区 |
|------|----------|-----------|
| `turn_runtime.py:_run_turn` | `trace_context(name="turn")` | 顶层 root run |
| `tool_registry.py:execute_tool` | `@safe_traceable(name="tool.execute")` | 工具调用+参数+结果 |
| `tool_registry.py:_execute_rag` | `@safe_traceable(name="rag.retrieve")` | RAG 检索结果 |
| `lightrag_engine.py:_llm_model_func` | `@safe_traceable(name="lightrag.llm")` | LightRAG 自建 client 盲区 |
| `llm.py` 主 client | `wrap_openai_client(..., "course_agent_chat")` | 主 LLM 调用 |
| `llm.py` fallback client | `wrap_openai_client(..., "course_agent_fallback")` | 兜底 LLM |
| `provider_factory.py` profile client | `wrap_openai_client(..., f"profile:{binding}")` | 运行期动态 client |

### Trace 层级（LangSmith UI）

```
turn (root) ← trace_context
├─ course_agent_chat (LLM) ← wrap_openai
├─ tool.execute (tool) ← @safe_traceable
│ └─ rag.retrieve (retriever) ← @safe_traceable
│ └─ lightrag.llm (llm) ← @safe_traceable
└─ course_agent_chat (LLM, 最终答案)
```

### 配置

```bash
# backend/.env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=course-agent-dev
```

### 成本核算与业务指标（Prometheus + 账单落库）

业务级指标定义在 `core/observability/metrics.py`（`ca_` 前缀），LLM 成本核算在 `core/observability/cost.py`。两类 LLM 成本指标命名对齐 OTel GenAI 语义约定：

| 指标 | 类型 | label | 埋点位置 |
|------|------|-------|---------|
| `ca_llm_tokens_total` | Counter | `model, token_type(input/output/cache_read)` | `_one_round` 逐轮（`cost.observe_usage`） |
| `ca_llm_cost_usd_total` | Counter | `model, course_id, mode` | `run_agent_loop` 按 loop 汇总（`cost.observe_cost`） |

**横切采集**：成本是横切关注点，落在 `_one_round`（四条 pipeline 全经过的唯一 LLM 出口）而非各 capability。token 维度逐轮埋（有 `model`）；cost 维度按 loop 汇总埋（`course_id`/`mode` 需 `context`，故在 loop 层而非 `_one_round`）。

**跨 provider usage 归一**：OpenAI 开 `stream_options={"include_usage":True}` 后，末块 `choices=[]` 携带 `usage`；Anthropic 适配器（`anthropic_adapter._run`）从 `message_start`（input/cache_read）+ `message_delta`（output）攒齐后，在 `message_stop` 合成**同形态** usage 块。`cost.usage_from_response_chunk` 一套代码读两类 provider。`agent_loop.done` / `turn.complete` 日志带 `input_tokens/output_tokens/cache_read_tokens/cost_usd`，可算 cache 命中率 = `cache_read/input`。

**价目表**：`data/model_pricing.json`（每 1M token 美元价，input/output/cache_read 三档），与 `model_catalog.json` 同目录，按 mtime 热重载；支持家族前缀模糊匹配（`deepseek-v4-pro` 命中 `deepseek-v4`）。`estimate_cost` 命中算价，未命中/无价目表/usage 为 None 返回 0（可观测但不阻塞）。

**成本配额（软降级，第四批）**：`core/quota/cost_quota.py` 按 **user × course × 自然日** 在 Redis 计数（键 `ca:costquota:{user}:{course}:{YYYYMMDD}`，TTL 2 天自清理）。`loop.py` 在 `estimate_cost` 后 `accrue_cost` 累加；`chat_pipeline` 解析 runtime 后 `check_quota`，超 `settings.cost_quota.daily_budget_usd` 时把模型从 `text_model` 降到便宜档 `fast_model`（`dataclasses.replace`，**只降级不拒绝**——避免「上一轮刚好超限→下一轮被锁死」）。默认 `enabled=False`（行为零变化）；Redis 不可用时 check 放行、accrue 静默跳过，绝不阻塞业务。软滞后：成本 loop 结束后才累加，故「推过预算的那一轮」本身不降级，下一轮才降级（软限流固有特性，非 bug）。

**stream_options 安全降级**：不支持的 OpenAI 兼容端点会对 `stream_options` 报 400 而非静默忽略。`llm._create_with_image_fallback` 内捕获该错误后剥掉 `stream_options` 重试一次（与 Stage-2 剥图降级同套路，发生在 create 层、reliability 之前，丢本轮 usage 但保 turn 不挂）。

**用量持久化（账单级，落库）**：Prometheus Counter 无法按 `user_id` 展开（label 基数爆炸，Prometheus 硬约束），Redis 配额键只留 2 天，故「按人/按课查账」另落 PG。`core/analytics/token_usage.py` 两级存储：`llm_usage_records`（明细，每 `run_agent_loop` 一行，无 FK 抗级联删，带 `mode`/`rounds` 供 cost-of-pass 分析，`cost_usd` 摄取时按价目表快照落库——日后改价不篡改历史账）+ `llm_usage_daily`（日聚合读模型）。`loop.py` 在 `observe_cost`/`accrue_cost` 旁 `record_llm_usage`（`usage_tracking.enabled` 默认 True，best-effort 不阻塞）；ARQ `cron_rollup_usage` 每小时 :11 删今日+昨日聚合行后重插（幂等）+ 清过期明细（`detail_retention_days`，日汇总永久保留）。查询 `GET /admin/usage/summary`（全量，group_by day|user|course|model）与 `GET /teacher/courses/{id}/analytics/usage`（`_get_owned_kb` 归属隔离，group_by day|user|model）只读 daily 表。`done` 事件 `metadata.usage` 透出本轮 token（学生端气泡底部小字；美元成本仅 `expose_cost_to_student` 开启时带）。

---

## LLM Provider Catalog (per-request, )

The backend supports **per-request provider/model switching** admins pre-configure a pool of LLM provider profiles (`data/model_catalog.json` → `profiles[]`), and any user can pick one from a dropdown in the chat window for that turn — **no restart, takes effect immediately**.

```
ChatWindow (model dropdown) ──► POST /api/chat { model_profile_id }
 └── UnifiedContext.llm_profile_id
 └── pipeline_common.resolve_profile_runtime
 ├── core/llm/catalog.get_profile_cached(id) # Redis cache-aside（miss→to_thread 读 JSON）
 └── provider_factory.get_llm_client_for_profile # fingerprint-cached client
 └── run_agent_loop(client=<profile>, model=<profile.text>) # loop already accepts client/model
```

Key points:
- **`run_agent_loop`'s core loop is unchanged** — it already accepted a `client` override (`loop.py:214/258`); this is a clean injection path, not a rewrite of the scheduling kernel.
- **Empty fields fall back to `.env`** (`get_llm_client_for_profile`): `active=default` (catalog key/base_url usually empty) still constructs correctly via the startup constants — so "no selection" and "select active profile" behave identically.
- **Fingerprint-cached clients**: same `(binding, key, base_url, api_version)` reuses one client (avoids per-request `new`).
- **No selection → uses catalog `active_profile`**. 运行期读走 `load_catalog_cached()`（Redis cache-aside，TTL 300s 纯兜底）；admin 写端点（`upsert`/`delete`/`set_active`）调 `invalidate_catalog_cache()` 显式失效 → 下一个请求全部 4 个 worker 都看到新值（无重启、无"幽灵 profile"）。复用 `core/db/cache.py`，与 `get_course_prompt` 同构。
- **Scope**: only the chat / deep_solve turn is per-request switchable (both route through `/api/chat` → ChatPipeline). Embedding / LightRAG / Bot subsystems keep using the startup constants — .

Management API (`api/llm.py`, admin): CRUD profiles + `/probe` test connection + `/active` default; `/profiles/selectable` feeds the dropdown (**api_key stripped**). Catalog read/write layer: `core/llm/catalog.py` —— 同步 `load_catalog/get_profile`（启动期 / 单测用）+ async `*_cached`（运行期热路径：Redis cache-aside + `asyncio.to_thread` 读文件，避免同步 I/O 阻塞 worker 事件循环；写后 `invalidate_catalog_cache` 显式失效）。写端点用 `asyncio.to_thread` 包文件写。Frontend: `LlmProviderPage`（admin CRUD+test；学生视图可自配个人 Key 覆盖平台默认、走自己额度，支持「复原为默认」回退平台共享模型）+ `ChatWindow` model dropdown (all users).

## RAG 检索架构（`core/rag/`）

四阶段递进检索，每层独立开关，**默认配置下行为与改造前完全一致**（ES / 上下文分块均默认关），逐层启用可量化增益：

**自适应路由（`tool_registry._execute_rag` + rag schema `strategy`）**：
- `fact`（默认）→ `retrieve_context(mode="naive")` 纯向量精确匹配；ES 启用时 `retrieve_context` 内部自动走 hybrid。
- `relationship` → `graph_augmented_retrieve`：local 图谱邻域（实体中心 1-hop 扩展，解释多跳关系）+ naive 向量事实证据，去重拼接。比 mix 更聚焦（去掉 global 全局摘要噪声）。

**相关性三道防线（`lightrag.py` + `hybrid_retriever.py` + `tool_registry._execute_rag`）**：检索召回后串三道闸防「无命中被当成证据」。第 1、2 层各后端各自实现且**默认不常开**（第 1 层按后端命中形态不同、第 2 层需精排开启），**只有第 3 层两后端共享且常开**：
1. **无命中判定（各后端各自实现）**：LightRAG naive_query 无 chunk 时返回 None -> `aquery_llm` 用 `fail_response` 兜底，产出 `"Sorry, I'm not able to provide an answer to that question.[no-context]"` 非空字符串，`_extract_contexts` 在**单一 chokepoint** 匹配 `[no-context]` 标记拦截（覆盖 retrieve/retrieve_context/_retrieve_hybrid/dense_search/graph_augmented_retrieve/query 全部 6 条路径）命中返回 `[]`。pgvector 路无此病--无命中即空 list，`retrieve_context` 直接 `return ""`，不需哨兵。
2. **相关性阈值过滤（两后端同口径，默认关）**：低于阈值的 chunk 在 rerank 后被丢弃。LightRAG 走原生 `settings.lightrag.min_rerank_score`（经 `instance_pool` 签名注入）；pgvector 走 `settings.rerank.min_score` -> `llamaindex_pg._fuse` 经 `replace(...)` 注入 `RetrievalConfig.min_rerank_score` -> `hybrid_retriever.retrieve` 精排成功后过滤。**两后端同为 qwen3-rerank 的 `relevance_score`，同一数值可共用**。默认 0.0=不过滤=行为不变；仅在精排开启（LightRAG `enable_rerank=True` / pg `RERANK__ENABLED=true`）且拿到 `rerank_score` 时生效。阈值放在 `hybrid_retriever` 的 `try` 内 `fused=reranked` 之后：精排 API 失败降级回融合结果时无 `rerank_score`，阈值不生效（不把一次 API 抖动变成全课程拒答）；未被精排打分的 leftover/tail 一并丢弃（与 LlamaIndex `SimilarityPostprocessor` 丢 `score=None` 同语义）。**不可照搬裸余弦/RRF 分数的 0.5**--cross-encoder 分数跨 query 不可比，阈值须在本课程含负样本的留出集上标定（见 plan 的 risk-coverage 调研）。
3. **工具层无命中信号（两后端共享，唯一常开）**：`_execute_rag` 空 content（含被哨兵拦截的、或被阈值全滤掉的）返回 `success=False` + 中文无命中提示且**不带 sources**，让 Agent 转而说明「课程资料未覆盖」而非用通用知识硬答。`rag.yaml` 的 note 同步该语义。阈值把结果全滤掉时 -> `results` 为空 -> `retrieve_context` 返回 `""` -> 本层自动拒答，第 2、3 层天然串起，无需新增管道。



**切块策略（`core/rag/chunking/` + `ingestion.parse_files`）**：`settings.chunking.strategy` 是可插拔扩展点，由 `core/rag/chunking/registry.py` 注册表分发（`register_chunk_strategy`/`get_chunk_strategy`，与检索器后端 `core/rag/registry.py` 同构；新增策略 register 一行，不改 `ingestion._chunk_documents` 的分发代码）。默认 `sentence_splitter`（LlamaIndex SentenceSplitter 按字数切，行为与改造前完全一致，兼作消融对照组）。`ragflow_manual_docx`：DOCX 走移植自 RAGFlow Manual 的结构化切块——**标题层级栈**（`Heading N` 维护父子栈，每块带祖先标题路径，比纯按字数切上下文更全；背书见 MultiDocFusion arXiv:2604.12352，层级感知切块 retrieval precision +8–15%）+ **表格原子化**（整表作为独立块，不进文本切分，治 SentenceSplitter 把大表格从 `|` 处锯断的截断 bug）；非 DOCX（PDF/TXT/PPTX）仍回退 sentence_splitter。两路产出都经 `_build_source_prefix` 注入 `【章节:】`/`::chunk-<idx>`，下游 `_ingest_body` 零改动。零新依赖（python-docx + SentenceSplitter 兜底）。PDF 的结构化（章节/表格/页码）由 loader 层 `extract_pdf_sections` 完成（见下「PDF 摄入」），切块仍走 sentence_splitter——不再有 pdf_structured 策略。

**PDF 摄入（`file_routing.extract_pdf_sections` + `indexing_documents`，backend 二选一）**：PDF 与 DOCX 同构——loader 层 `extract_pdf_sections` 把 PDF 装成 `[{title, content, page}]` section 列表（与 `extract_docx_sections` 同款），每个 section 转一个带 `section`+`page` metadata 的 LlamaIndex Document，切块统一走 sentence_splitter（不再有 pdf_structured 策略）。治旧 `extract_pdf_text`（仅一句 `page.get_text()` 按内部顺序拼文本）的三病——表格腰斩、多栏错乱、无页码/章节。`settings.pdf.backend` 二选一（无降级链，选定失败则跳过该文件，不切另一个）：
- **docling（默认）**：Docling 单引擎全包——版面（DocLayNet）+ 表格（TableFormer）+ 标题层级 + 阅读顺序 + 页码 provenance + 扫描件 OCR（RapidOCR 后端，PaddleOCR 模型 ONNX 版，免装 PaddlePaddle 重框架）。`label=section_header` 开新 section，`label=table` 表格原子化为独立 section（不锯断）。模块级 `_docling_converter` 单例 + `threading.Lock` 守护首次加载（arq worker max_jobs=10 并发下只加载一次）。
- **mupdf**：PyMuPDF `get_toc()` 切章节 + `page.get_text()` + `page.number` 注入页码，轻量纯 CPU（无 torch）；无 OCR、无表格原子化——轻量 trade-off。
产出经 `_build_source_prefix` 注入 `【章节: x | 第N页】`（`_chunk_by_sentence_splitter` 读 `node.metadata['page']`），下游 LightRAG `ainsert` / ES 双写 / contextual_enrichment / 图片KG 全部零改动。背书：Docling arXiv:2408.09869、OmniDocBench CVPR 2025（表格 cell 准确率 ~97.9%）、RapidOCR。

**上下文分块（Phase 2，`contextual_chunking.py` + `ingestion`）**：`chunking.contextual_enrichment=true` 时，ingestion 为每个 chunk 用 fast LLM 生成「该 chunk 在全文的位置/主题」前缀（Anthropic 上下文检索，召回失败率 −49%）。按源文件分组、md5 缓存、单块失败降级原文。默认关（改了要重新索引）。

**ES 混合搜索（Phase 3，默认关）**：
- `es_client.py` — ES BM25 chunk 索引（ik_smart 中文分词），惰性建连，未装/未连通时 `_ensure()=False` 安全降级。`chunk_id=md5(content)` 作 LightRAG↔ES join key。
- `hybrid_retriever.py` — config 驱动：BM25(ES)+Dense(LightRAG naive) 并行召回 → RRF/linear 融合 → 可选 rerank。任一路失败跳过该路。
- `retrieval_config.py` — `RetrievalConfig` 旋钮 + `ABLATION_CONFIGS` 7 组消融预设（dense_only/bm25_only/hybrid_rrf+rerank…）。
- 接入：ingestion 双写 ES（`_index_batch_to_es`）；`retrieve_context` naive 路径在 ES 启用时走 hybrid，否则原 naive。dense 路复用 LightRAG naive（内部已 rerank），故 hybrid 层不再外挂 rerank_fn 避免双重精排。

**启用 ES**：`.env` 设 `ELASTICSEARCH__ENABLED=true` + `ELASTICSEARCH__URL`（docker 网络用服务名 `http://elasticsearch:9200`），装 ik 插件（见 `docker-compose.yml` 注释）。

**索引后端（per-KB，`core/rag/registry.py` + `KnowledgeBase.index_backend`）**：一门课可**同时构建 LightRAG 与 pgvector 两套索引**（双索引，状态存 `kb_builds` 子表），问答时按 `rag_mode` 路由（`tool_registry._execute_rag`）：`llamaindex_pg`/`mix`/`naive`/`local`/`global` 显式指定（`llamaindex_pg` 强制走 pg 向量，后三者走 LightRAG 原生模式），`auto`（默认）按 Agent `strategy` 自动选（relationship→图谱、fact→pg 向量）；也兼容只建一个后端的老课程（迁移自动回填一条 `kb_builds`，行为等价旧单后端）：
- `lightrag`（默认）：知识图谱，逐 chunk LLM 实体/关系抽取（慢，数千 chunk 即数小时），但支持多跳关系推理（relationship 路径走 `graph_augmented_retrieve` 图邻域扩展）。
  - 存储后端（`instance_pool._get_instance`）：Postgres 部署把 KV/Vector/DocStatus 三类存储从进程内存搬到 Postgres（`PGKVStorage`/`PGVectorStorage`/`PGDocStatusStorage`，按 `workspace=course_{id}` 列隔离，LIGHTRAG_* 表 `IF NOT EXISTS` 幂等建），把每门课常驻 RSS 从数百 MB 降到数十 MB；图谱显式保留 `NetworkXStorage`（文件 graphml，体积小；不上 AGE/Neo4j，前者建图实测 434s 灾难、后者超 4GB 内存预算）。SQLite 部署（本地/测试）保持默认内存后端——由 `_ensure_lightrag_pg_env()` 把 `settings.db.url` 桥接成 LightRAG 认的 `POSTGRES_*` env（`MAX_CONNECTIONS=5`/进程；绝不设 `POSTGRES_WORKSPACE`，否则覆盖 per-course workspace 致多租户混串）。`purge_course_workspace` 重索引前对 11 个 PG storage 调 `drop()`（`DELETE WHERE workspace=$1`，不 DROP 共享表）+ rmtree graphml。
- `llamaindex_pg`（pgvector 快速向量）：给"量大、只需事实检索、受不了几小时索引"的课。复用 `ingestion.parse_files` 切好的 chunks → embedding 批调用 → 存 Postgres（`PGVectorStore` + HNSW + tsvector），分钟级建索引，进程不常驻（治旧 `SimpleVectorStore` 落 JSON 的三病：暴力扫描无 ANN、每 worker 常驻一份、索引文件与 PG 分家）。摄入时 `persist_ingest_chunks(backend="llamaindex_pg")` 会落一份 `data/ingest_chunks/course_{id}/latest.json`——这是**审计副本**（含全量 chunks + `node_ids`，供排查时 join 回 `data_kb_chunks` 主键行），与检索无关：向量仍只在 PG，不是回退到旧 JSON 索引；与 LightRAG 侧 `lightrag_store/.../ingest_chunks/` 共用同一 writer，便于对照两后端吃到的 chunk 是否一致。
  - 检索绕开 PGVectorStore 无融合的 hybrid 模式（它只去重合并、无 RRF、alpha 被忽略，见 run-llama/llama_index Discussion #19606）：dense（DEFAULT 走 HNSW）+ sparse（PG tsvector 全文，zhparser 按词切分）各查一次 → 交 `hybrid_retriever` 做 RRF 融合。两路查同一张表 `data_kb_chunks`，chunk_id（node_id）天然一致，无需像 LightRAG+ES 双写对齐。
  - 精排（可选，默认关）：`_fuse` 经 `core/rag/rerank.py::build_rerank_fn` 注入 DashScope qwen3-rerank（`RERANK__ENABLED`，凭证复用 `EMBEDDING__API_KEY`），对 RRF 融合结果做候选池截断（`candidate_top_n=20`）→ Cross-Encoder 精排（`rerank_top_n` 跟随调用方 `top_k`）。无 key / 关闭时返回 None、行为同无精排；失败由 hybrid_retriever 降级回 RRF 排序。召回档 `bm25_top_k/dense_top_k=20`（融合后≤40）是按精排 token 成本定的（见 plan 调研），可调。
  - 存储层 `core/rag/llamaindex/pg_store.py`：PGVectorStore 单例 + 官方 `OpenAIEmbedding` 接 `settings.embedding`（DashScope text-embedding-v3）；`perform_setup=True` 自动建表+HNSW（幂等），`course_id` 用 metadata filter 隔离（`indexed_metadata_keys` 建索引避免全表扫）。`text_search_config='chinese'`（zhparser 中文分词；`simple` 不分词会让整句压成一个 token、sparse 路 0 召回）。sparse 查询侧 `_PgSparseStore.bm25_search` **绕开库的 SPARSE 模式**（llama-index `base.py:956` 用 `to_tsquery` 且其预处理 `re.sub(r'\W+',' ',q)` 对中文不分词，整串一个 token），改直接 SQL：`unnest(to_tsvector('chinese',q))` 取 lexeme 用 `|` 连成 OR tsquery -> `ts_rank` 排序，与文档侧 `text_search_tsv` 同源分词、token 对齐。zhparser 默认 `dict_in_memory=off`，词典走 mmap 共享 OS page cache，per-connection 内存趋近 0（适配 4GB 部署）。
  - 构建分流（per-backend，写 `kb_builds` 行）：`trigger_kb_indexing(db,kb,course_id,backend)` → ARQ `run_indexing(...,backend)`，分布式锁 key 含 backend（`{course_id}:{backend}`）→ 两后端可并行构建不互斥；`_run_indexing`（lightrag）与 `_run_indexing_llamaindex_pg` 各写自己的 `kb_builds` 终态（KB 行旧 status/progress 列保留但不再被写，改由 `_kb_to_dict`/`_kb_to_course` 从 `kb_builds` 聚合派生，`lazy="selectin"` 自动连表）。就绪判定三处同口径：`status=='ready'` 且 `chunks_total>0`（`aggregate_build_status` 徽章 / `ready_backends` 按钮启用 / `_get_ready_backends` 检索路由）；0-chunk（解析全失败/空文档）的 build 终态判 `error` 并把解析失败原因（透传自 `parse_files` 的 `parse_errors`）写进 `error_msg`，杜绝空索引被误判就绪、徽章绿而按钮禁用。建库/索引 API（admin/teacher `index`/`pause`/`stop`）接受 `backend` 参数。
  - 检索路由（`tool_registry._execute_rag`，修了 `mode="naive"` 硬编码 bug——前端 `rag_mode` 经 `tool_dispatch` 注入的 `mode` 此前被吃掉从未生效）：`mode`=用户每请求选（`auto` 默认 / `mix`/`naive`/`local` 手动）；`strategy`=LLM 按 `rag.yaml` 自选（`fact`/`relationship`），两者正交。手动模式直接透传 lightrag `retrieve_context(mode=)`；`auto`：`relationship` + lightrag 就绪 → `graph_augmented_retrieve`（多跳），否则优先 pg 向量，再退 lightrag naive。`_get_ready_backends(course_id)` 查 `kb_builds` 就绪集合驱动路由。
  - 迁移：`016_kb_index_backend` 加 `index_backend` 列（存量回填 `lightrag`）；`018_kb_builds` 加 `kb_builds` 子表（`UNIQUE(kb_id,backend)`，从存量 KB 行回填一条）。向量表 schema 由 PGVectorStore 自动维护（不手写 migration 对齐其内部结构，避免版本耦合）；`030_zhparser_chinese_fts` 是例外--重建 `text_search_tsv` 列的 generated 表达式从 `to_tsvector('simple',text)` 改 `to_tsvector('chinese',text)`（装 zhparser 扩展，见 `postgres/Dockerfile`）。

**解析层（第二期，`core/rag/parsing/`，opt-in）**：文档解析独立成层，默认引擎 MinerU 托管 API（替代 worker 内 Docling，去 torch，稳态内存 ~2.5GB→~0.4GB——省下的给 LightRAG）。借鉴 DeepTutor `services/parsing/`：ParsedDocument IR（markdown+blocks）+ 内容寻址缓存（键=字节 sha256+parser_signature，`manifest.json` 最后写=ready）+ Parser Protocol + Service 调度（格式不支持直接报错不换引擎，单引擎哲学）。`engines/mineru_api.py` 同步 httpx 四步（POST file-urls/batch→PUT 上传→轮询 extract-results→下载 zip 解包，业务码双层检查+zip Slip/bomb 防护）。**大 PDF 分片**：页数超 `settings.parsing.max_file_pages`（默认 200）时引擎内部自动分片——fitz 按页等切子 PDF（不重叠）→ 逐片顺序走四步 → 按页序合并 markdown / content_list（各 block 的 `page_idx` 加该片起始偏移还原全局页码）/ images（按内容哈希命名跨片幂等覆盖）→ 清临时目录；上游缓存键=原文件字节 hash、service/ingestion 只见一个完整 ParsedDocument，分片是引擎内部实现细节。加密/损坏读不出页数则整文件直递（让 MinerU 自己拒）。**opt-in**：`settings.parsing.engine` 空=走原 file_routing（docling/mupdf，行为零变化），配 `mineru_api`=换托管 API。docling 移到 `[parse-docling]` extra（云端默认不装 torch）。`KBFile.parser_engine` 列（事后归因）。

**markdown 分节（MinerU PDF 主路径，`parsing/markdown_sections.py`）**：MinerU 摄入走 DeepTutor 式 markdown 路线——`indexing_documents` 的 `_parsing_engine` 分支不再调 `ParsedDocument.to_sections()`（= `blocks_to_sections`），改调 `markdown_to_sections(parsed.markdown)`：`full.md` → LlamaIndex `MarkdownNodeParser`（按标题分节，大小仍交下游 `SentenceSplitter`，不在解析层预写 chunk）+ `resolve_image_refs`（图片 `![](images/…)` inline 成文本，默认删 marker 留 `Figure/图` 图注，无图注才 fallback VLM，pgvector 无多模态 embedding 故必须 inline）。`settings.parsing.image_vlm_always`（默认关）开启后有图注的图也调 VLM，在 marker 位置插 `[图: 描述]`、图注保留——让「图 1-1 股权架构图」这类纯标题图里的结构/数字可被向量召回；先 `_prefetch_descriptions` 按 sha256(bytes) 去重 + `gather`+`Semaphore(image_ingest.semaphore)` 并发预取（181 张图不再串行 10 分钟），desc_cache 持久化、重复索引不重花。**根因**：MinerU 标题块是 `type=text`+`text_level`（非 `type=title`），`blocks_to_sections` 分不出节、整篇 PDF 拼成单 section；markdown 路线直接吃 `full.md` 的 `#`/`##` 层级才是正确分节（实测 MinerU `full.md` 已干净，0 条 header/footnote 混入）。**注意**：`MarkdownNodeParser.header_path` 元数据只含祖先标题，叶子标题是节点内容首行 `#`——`title` 与参考文献过滤都从内容首行取叶子。`title` 取叶子标题（与 DOCX/PPTX/docling 各 extractor 一致）。`blocks_to_sections`/`to_sections` **保留**（解析层通用 blocks→sections IR，有独立单测覆盖；生产 mineru 路径已不走它）。`settings.parsing.drop_references`（默认开）过滤 References/参考文献/Bibliography 章节（省 embedding + 减检索噪声）；`load_ir` 显式排除 `*_content_list_v2.json`（v2 是 `list[list[dict]]`，误选形状错）。`settings.pdf.backend` 默认改 `mupdf`（PyMuPDF 核心依赖，开箱即用；docling 降为 opt-in）。

**第四期收尾**：
- `GET /admin/rag/engines`：索引后端（lightrag/llamaindex_pg）+ 解析引擎（mineru_api/docling）能力探测（托管看 api_key、自托管看 find_spec），给前端建库选择用。前端建库表单（TeacherPage/AdminPage CreateKBModal）加 index_backend select。
- **drop-es**：docker-compose 移除 elasticsearch 服务（省 512MB JVM）。混合检索的 sparse 路改用 PG tsvector + zhparser 中文分词（llamaindex_pg 绕开库 SPARSE 自写 OR tsquery；LightRAG 的 ES 混合 opt-in 默认关，`get_es_store` 连不上自动降级纯 dense）。

## RAG 评估系统（`scripts/eval_rag/`）

离线评测 LightRAG 各检索模式（naive/local/global/mix）与生产路径的检索 + 生成质量，用 RAGAS 0.4.3 量化，CI 质量门禁阻断回归。`--ablation` 模式遍历检索配置消融组合（见下）。

**模块组成**：
- `config.py` — DashScope LLM/embedding 配置、指标分层（tier1 检索 / tier2 生成 / tier3 鲁棒性+领域）、`QUALITY_GATES` 阈值。
- `rag_runner.py` — 两步查询：`retrieve()` 取上下文（纯检索，无 LLM）+ `query()` 取答案；延迟分阶段采集（retrieve_ms/query_ms）。`--production-parity` 走生产路径 `retrieve_context(naive, top_k=5)` 对齐 `_execute_rag`。
- `ablation_runner.py` — `--ablation` 模式：遍历 `core/rag/retrieval_config.ABLATION_CONFIGS` 全组合（dense/bm25/rerank/融合方式），每个 `RetrievalConfig` 跑 hybrid retrieve 对比检索召回质量（`context_precision`/`context_recall`，answer 留空省 LLM，隔离检索变量）；跳过质量门禁（配置对比分析，非达标门禁）。ES 未启用时 bm25 配置退化为纯 dense。
- `ragas_evaluator.py` — collections API（`llm_factory` → InstructorLLM），8 个指标：context_precision/recall、faithfulness、factual_correctness(f1)、noise_sensitivity、answer_relevancy、teaching_accuracy/safety（AspectCritic）。embed 子类显式注入 dimensions（DashScope text-embedding-v3 不传会致 answer_relevancy 内部 cosine 全 0）。
- `stats.py` — numpy 分布（P50/P90/P95/std）+ scipy Welch t-test 回归检测 + 历史基线 delta。
- `quality_gate.py` — `check_quality_gate(summary)` 按 `QUALITY_GATES` 检查（`<metric>_min/_max` 取所有 mode 最差值；`latency__<field>__<stat>` 取最大），不达标 `sys.exit(1)` 阻断 CI。本轮未评测的指标跳过，不误判。
- `report_generator.py` — CSV + Markdown 报告，附录含分布 / 延迟 / 历史 delta 表。
- `production_feedback.py` — 生产问笔回流（见下「Phase 6 诚实边界」）。

**关键修复（vs 旧实现）**：
- 上下文不再用 answer 自我证明（Bug 1）：contexts 独立来自 `retrieve()`，answer 来自 `query()`。
- embed 显式 dimensions（Bug 2）：否则 answer_relevancy 内部 cosine 全 0。
- 逐条真实分数（Bug 3）：来自同一次 evaluate 的 per-question scores，不再把整体均值复制 N 遍。
- `LLM_API_KEY` 优先读 `DASHSCOPE_API_KEY`（Bug 6），而非 embedding 专用 key。
- `LIGHTRAG__ENABLED` 用双下划线（Bug 4）：pydantic settings 嵌套字段必须双下划线。

**Phase 6 诚实边界**：`production_feedback.py` 从 `messages` 表导出真实 (question, answer) + 启发式初筛可疑低质量（过短 / 拒绝话术 / 错误标记），产出**无 ground_truth 的待标注候选池**（`datasets/v3_production_candidates.json`）。本项目**无点踩/差评功能，RAG 命中信号（empty/retrieved_chars）只打日志不落库**，故这是真实问答导出 + 零成本粗筛，**非用户负反馈闭环**；真正的闭环需要 messages.metadata 写入 RAG 命中字段 + 前端点踩按钮（独立功能，不在本脚本范围）。初筛顺序为 empty → refusal(语义) → too_short(长度)，确保短拒绝（"我不知道"）命中高置信 refusal 而非被长度截胡。

**测试**：`tests/test_ragas_eval_smoke.py`（14，mock LightRAG/RAGAS 纯逻辑 + 两步查询）+ `tests/test_production_feedback.py`（8，配对 / 初筛纯函数）。真实分数需在有 DashScope key 的环境跑 `python -m scripts.eval_rag.run_eval`。

## 记忆系统（L2/L3 四层架构，`core/memory/`）

四层职责（episodic / semantic / mastery / procedural）+ 事件驱动巩固。诊断与依据见 plan `l3记忆沉淀重构`。

```
热路径(零LLM)                后台巩固(慢模型)                   读路径(每轮)
CAPABILITY_COMPLETE ──► INSERT memory_episodes ──► consolidate_memory job ──► 注入 prompt
  (record_episode)        status=pending         segment→mem0.add         memory_context
  + importance 累计        +importance阈值/quiz   →merge graph+append       mastery_context
  超阈值→enqueue job       里程碑/cron兜底         mastery→procedural
```

- **L2 会话摘要**（`session_summary.py`，存 `sessions.summary` 单列 JSON）：v2 格式 `{"v":2,"items":[{k,key,t,ts,n}]}`，7 类 kind（topic/fact/decision/open_question/next_step/intent/misconception）。LLM 只抽【新增】并输出 slot key + resolved，代码用 `max(ts)` 做单值槽（fact/next_step）冲突裁决 + salience（`kind_weight×exp(-Δt/half_life)×(1+ln n)`，token 预算+每类上限）淘汰--替旧「精确去重 + `combined[-5:]` 硬截断」，治 P0 矛盾条目并存与「丢最旧而非最没用」。`ts` 代码盖章（增量末条消息 created_at），不让 LLM 出时间戳（arXiv:2606.01435）。v1 五键 dict 自动升级、旧自由文本透传。锁/OCC/keyset 增量逻辑不变。注入点 `chat.py`(SSE)+`run.py`(WS) 经 `get_summary` 渲染。评测 `scripts/eval_l2_summary/`（knowledge_update 基线 0% -> 新管线 100%）。
- **episodic**（`memory_episodes` 表，alembic 020）：原始 turn，永不删除，兼巩固 outbox。`(session_id,turn_id)` 唯一幂等；status pending/processing/done/dead。`record_episode`（`episodic.py`）在 EventBus `_on_capability_complete` 里写，importance 用零 LLM 启发式（长度/疑问/纠错/工具/模式）。
- **semantic**（mem0）：事实条目，由巩固 job 从 segment 升格（`mem0.add`，ADD-only 不覆盖——教育场景「上周不会这周会了」是状态演进不是冲突）。
- **mastery**（`knowledge_mastery` 表，alembic 021）：知识点掌握度，**带 course_id**（修 `users.knowledge_graph` 跨课程污染）。`append_mastery`（`mastery.py`）追加观测不覆盖（blend mastery/risk + count++ + evidence_episode_ids）；读时 `get_mastery_context` 按 `last_observed_at` 做 `exp(-λ·age)` 软衰减（半衰期 ~69 天，不物理删除——反复性错误证据留存）。
- **procedural**（personal SKILL.md）：巩固 job 攒够观测后生成学习画像草稿（`procedural.py`），写 personal 层，**不自动 always** + frontmatter `auto_generated:true` 待人工确认。

**巩固 job**（`consolidation.py` + `worker.consolidate_memory`）：claim pending（条件 UPDATE 并发安全）→ 按 session 分组（对齐 SeCom segment 级）→ 每组 mem0.add 升格 + 单次抽取喂 graph(dashboard)+mastery → 成功标 done / mem0 失败回 pending 重试。触发：热路径 importance 累计 ≥ 阈值(`mem0.consolidation_importance_threshold=0.7`) 或 quiz 里程碑 → enqueue；`cron_consolidate_memory`(5min) safety net 兜底长期 pending + 超时 processing。老 `cron_flush_memory` 降级 5min 排干旧 Redis buffer。

**回流 prompt**（`pipeline_common.build_common_context_layers`）：mastery_context 紧跟 memory_context 之后（L2 用户级易变段，**不破 prefix cache 前缀**），四条 pipeline(chat/solve/research/quiz) 共享此入口都拿到掌握度。

**评测**（`scripts/eval_memory/`）：照 LongMemEval 的 knowledge updates / abstention 维度，对 mastery 层做三维程序化判分（knowledge_update/abstention/decay，无需 LLM 可入 CI）。`python -m scripts.eval_memory.run`，门禁全过 exit 0。

**止血修复（Phase 1）**：`flush_manager._flush_turns` 返回 bool——mem0 关键写失败返回 False，`_flush_one` 保留 key 重试（修「写失败即永久丢数据」，原实现吞异常架空 H-7）；`graph_memory._turn_counter` 模块 dict 改 Redis INCR（跨 worker 共享）。注意：Phase 2 后 Redis buffer 不再被喂，老 flush 路径仅排干残留。

