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
| `core/agent/orchestrator.py` | `normalize_mode`（chat_mode 规范化）|
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
| quiz | `core/question/pipeline.py` | explore → plan (JSON templates) → quiz (per-question JSON + schema validation, 6 question types) |
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
| `web_search` | DuckDuckGo web search |
| `ask_user` | pause the loop to ask the user for clarification (WS entry only) |
| `solve_plan` / `solve_finish_step` / `solve_replan` | solve deterministic spine (deep_solve turn only; session_id injected via contextvar) |
| `read_skill` | dynamic schema — mounted when `context.skills_manifest` is non-empty; fetches a SKILL.md playbook |
| `load_tools` | dynamic schema — mounted when deferred MCP pool is non-empty; loads MCP tool schemas into the turn |
| `mcp_<server>_<tool>` | MCP server tools (deferred by default; registered into ToolRegistry, executed via `registry.execute`) |

The built-in tools (`rag`/`web_search`/`ask_user`/`solve_*`) are **context-gated**: `context.enabled_tools` controls which are mounted. The dynamic tools (`read_skill` / `load_tools` / `mcp_*`) are assembled by `DynamicToolResolver` (see below), not the static `TOOLS_OPENAI_SCHEMA`.

---

## Progressive Disclosure: Skills + Deferred MCP

Both **Skills** (knowledge playbooks) and **MCP tools** (deferred) reach the model the same way — the system prompt carries a **one-line manifest** per item, and the model fetches the full content on demand. This keeps the always-on schema surface small (improves tool selection on weaker models) while keeping every skill/tool one cheap call away.

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

## RAG 评估系统（`scripts/eval_rag/`）

离线评测 LightRAG 各检索模式（naive/local/global/mix）与生产路径的检索 + 生成质量，用 RAGAS 0.4.3 量化，CI 质量门禁阻断回归。

**模块组成**：
- `config.py` — DashScope LLM/embedding 配置、指标分层（tier1 检索 / tier2 生成 / tier3 鲁棒性+领域）、`QUALITY_GATES` 阈值。
- `rag_runner.py` — 两步查询：`retrieve()` 取上下文（纯检索，无 LLM）+ `query()` 取答案；延迟分阶段采集（retrieve_ms/query_ms）。`--production-parity` 走生产路径 `retrieve_context(naive, top_k=5)` 对齐 `_execute_rag`。
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
