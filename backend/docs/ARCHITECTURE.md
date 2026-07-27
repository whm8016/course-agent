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

### 上下文预算治理（in-loop context budget）

多轮工具调用会让 `messages` 滚雪球撑爆 context window（complexity trap）。`core/agentic/context_policy.py` 在每轮 LLM 调用前做三段式裁剪，`settings.context_policy.enabled=False`（默认）时 loop 仍走旧 `_snip_tool_results`（按全局字符 > 80000 从最早 tool 替换），**行为零变化**，便于做 raw vs masking 消融对照：

| 函数 | 作用 | 调研依据 |
|------|------|---------|
| `cap_tool_result` | 单条工具结果**段落边界感知截断**（按空行切段、最后一段在句号处截断、尾部「来源」段优先保留）；单段无边界退化为头尾各半 | RECOMP（arXiv:2310.04408）extractive compressor：按语义边界选内容不破坏句子；消除「前端截 2000 字、messages 收全量」的不对称 |
| `mask_old_observations` | 按 **assistant-with-tool_calls 边界**切轮，保留最近 M 轮 tool 原文，更早替换占位（**按轮不按全局字符**） | arXiv:2508.21433（JetBrains，SWE-bench Verified）：Observation Masking 相对 raw 成本减半、解题率持平 |
| `apply`（hybrid 兜底） | 被掩码轮数 ≥ 阈值时，用 fast LLM 把被掩码原文压成摘要塞回原位（结构不变，仅改 tool content，不破坏 `tool_call_id` 匹配） | 论文：纯摘要引发 trajectory elongation，最优是 hybrid |

**评测四臂**（contextvar `set_arm`/`apply_arm`，对照 arXiv:2508.21433）：`raw`（真基线，不裁）/ `masking`（按轮掩码）/ `summary_only`（每轮摘要，测 H2 elongation）/ `hybrid`（掩码为主+摘要兜底，论文最优）。arm 经 contextvar 注入，串行评测 set→跑→reset 不污染下一组合，生产代码零侵入。消融 runner 见 `scripts/eval_context/`（`run_ablation` 量化各臂 token 成本/轮数/额外压缩调用）。

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
 ├── assemble system_prompt (chat.yaml loop spec + course prompt + persona + memory)
 └── run_agent_loop(...)
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
| MCP Connection ×4 | stdio 子进程 ×4 | Leader 选举，仅 leader 启动 MCP |
| TurnRuntime KeyError | SSE 断线重连路由到不同 worker → `_executions` 找不到 | Nginx `ip_hash` 粘性会话 |
| Circuit Breaker 语义减弱 | 每个进程独立 failure_threshold → 总阈值放大 N 倍 | 参数缩放：`failure_threshold // workers`（最小 2） |
| LightRAG LRU 缓存 | 4 份缓存，内存占用 ×4 | 容量缩放：`capacity // workers` |

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

### 成本核算与业务指标（Prometheus）

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

---

## LLM Provider Catalog (per-request, )

The backend supports **per-request provider/model switching** admins pre-configure a pool of LLM provider profiles (`data/model_catalog.json` → `profiles[]`), and any user can pick one from a dropdown in the chat window for that turn — **no restart, takes effect immediately**.

```
ChatWindow (model dropdown) ──► POST /api/chat { model_profile_id }
 └── UnifiedContext.llm_profile_id
 └── ChatPipeline._resolve_profile_runtime
 ├── core/llm/catalog.get_profile(id) # read JSON (live)
 └── provider_factory.get_llm_client_for_profile # fingerprint-cached client
 └── run_agent_loop(client=<profile>, model=<profile.text>) # loop already accepts client/model
```

Key points:
- **`run_agent_loop`'s core loop is unchanged** — it already accepted a `client` override (`loop.py:214/258`); this is a clean injection path, not a rewrite of the scheduling kernel.
- **Empty fields fall back to `.env`** (`get_llm_client_for_profile`): `active=default` (catalog key/base_url usually empty) still constructs correctly via the startup constants — so "no selection" and "select active profile" behave identically.
- **Fingerprint-cached clients**: same `(binding, key, base_url, api_version)` reuses one client (avoids per-request `new`).
- **No selection → uses catalog `active_profile` (read live)**, so `set_active` also takes effect immediately (not just post-restart).
- **Scope**: only the chat / deep_solve turn is per-request switchable (both route through `/api/chat` → ChatPipeline). Embedding / LightRAG / Bot subsystems keep using the startup constants — .

Management API (`api/llm.py`, admin): CRUD profiles + `/probe` test connection + `/active` default; `/profiles/selectable` feeds the dropdown (**api_key stripped**). Catalog read/write layer: `core/llm/catalog.py` (live file reads → API writes visible next request). Frontend: `LlmProviderPage` (admin CRUD+test) + `ChatWindow` model dropdown (all users).

## RAG 检索架构（`core/rag/`）

四阶段递进检索，每层独立开关，**默认配置下行为与改造前完全一致**（ES / 上下文分块均默认关），逐层启用可量化增益：

**自适应路由（`tool_registry._execute_rag` + rag schema `strategy`）**：
- `fact`（默认）→ `retrieve_context(mode="naive")` 纯向量精确匹配；ES 启用时 `retrieve_context` 内部自动走 hybrid。
- `relationship` → `graph_augmented_retrieve`：local 图谱邻域（实体中心 1-hop 扩展，解释多跳关系）+ naive 向量事实证据，去重拼接。比 mix 更聚焦（去掉 global 全局摘要噪声）。

**相关性三道防线（`lightrag.py` + `tool_registry._execute_rag`）**：检索召回后串三道闸防「无命中被当成证据」：
1. **无命中哨兵拦截**：LightRAG naive_query 无 chunk 时返回 None → `aquery_llm` 用 `fail_response` 兜底，产出 `"Sorry, I'm not able to provide an answer to that question.[no-context]"` 非空字符串。`_extract_contexts` 在**单一 chokepoint** 匹配 `[no-context]` 标记拦截（覆盖 retrieve/retrieve_context/_retrieve_hybrid/dense_search/graph_augmented_retrieve/query 全部 6 条路径），命中返回 `[]`——否则会被包成 `[证据1]` 喂给 Agent + 带 sources 弹来源卡片。
2. **相关性阈值过滤**：`settings.lightrag.min_rerank_score`（默认 0.0=不过滤=行为不变）经 `instance_pool` 签名注入 LightRAG 原生 `min_rerank_score`（1.5.4+），低于阈值的 chunk 在 rerank 后被丢弃。仅在 `enable_rerank=True` 且 `rerank_model_func` 已挂载（需 `EMBEDDING__API_KEY`）时生效；缺 key 静默失效，但哨兵修复不受影响。口径是 gte-rerank-v2 的 `relevance_score`，不可照搬裸余弦 0.5。
3. **工具层无命中信号**：`_execute_rag` 空 content（含被哨兵拦截的）返回 `success=False` + 中文无命中提示且**不带 sources**，让 Agent 转而说明「课程资料未覆盖」而非用通用知识硬答。`rag.yaml` 的 note 同步该语义。



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
