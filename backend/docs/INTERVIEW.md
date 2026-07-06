# 课程Agent — 面试讲解稿

> 面试不是讲"我写了多少代码"，而是讲清"我做了什么、为什么这么做、取舍是什么"。
> 本文档按"一句话定位 → 架构 → 数据流 → 亮点 → 取舍 → 差异化 → 简历话术"组织，每节都应能脱稿讲。
> 所有引用都带 `file:line`，可随时回代码核对。

---

## 0. 一句话定位（30 秒）

> "我做了一个**多租户的课程 AI SaaS**：教师端备课、建知识库、看学情分析；学生端基于课程知识库做智能答疑、解题、出题。架构对标开源单机版的 DeepTutor，但我把它做成了**真正的多租户 SaaS**——加了 RBAC 权限、教师后台、学情看板、异步任务队列。"

一句话说清三件事：**是什么**（课程 AI SaaS）、**对标谁**（DeepTutor）、**差异化**（多租户 SaaS vs 单机库）。

---

## 1. 技术栈（30 秒扫一遍）

| 层 | 选型 |
|---|---|
| Web | FastAPI（async）+ Gunicorn 4 worker |
| 持久化 | PostgreSQL（SQLAlchemy 2.0 async）+ Redis |
| 任务队列 | ARQ（索引 / 解题 / 研究 / 定时总结跑在独立 worker） |
| RAG | LightRAG（图谱+向量混合）+ LlamaIndex 双引擎 |
| LLM | 14 provider 抽象（OpenAI/DeepSeek/通义/GLM/Kimi/Claude…），熔断器 + fallback |
| Agent | 自研 **tool_calls 驱动** Agent Loop + 单层 Capability（四能力对齐 DeepTutor） |
| 流式 | SSE + WebSocket 双通道，StreamBus fan-out |
| 前端 | React 19 + TS + Vite + Tailwind |

---

## 2. 架构图（要能手画）

```
┌─────────────────────────────────────────────────────────────┐
│  前端 SPA（React 19 + TS）                                    │
│  学生端 ChatWindow │ 教师端 TeacherPage │ 管理员 AdminPage    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE / WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│  API 网关（FastAPI）                                          │
│  JWT 鉴权 │ SlowAPI 限流 │ check_course_access 多租户隔离     │
│  POST /api/chat(SSE) │ WS /api/run/{capability}              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  编排层                                                       │
│  CourseOrchestrator → 按 mode 选 Capability                   │
│  Capability: chat / deep_solve / deep_research / quiz / ...   │
│  → 统一 Agent Loop（tool_calls 多轮）                         │
└──────────┬───────────────────────────────┬──────────────────┘
           │                               │
┌──────────▼──────────┐         ┌──────────▼──────────────────┐
│  工具面              │         │  双引擎 RAG                  │
│  rag │ web_search    │         │  LightRAG(图谱+向量)         │
│  ask_user(可暂停)    │         │  LlamaIndex(多模态文档)      │
└──────────────────────┘         └─────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│  LLM 抽象层（provider 无关）                                  │
│  AsyncOpenAI / AnthropicAdapter / AsyncAzureOpenAI           │
│  熔断器 + 指数退避 + fallback client + 并发信号量             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  基础设施                                                     │
│  PostgreSQL（用户/会话/KB/enrollments）│ Redis（缓存/队列）    │
│  ARQ Worker（后台索引/解题/研究）│ LightRAG Store（每课独立）  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 一条完整数据流（学生提问 → 流式回答）

> 这是面试最常问的"讲一个请求的完整链路"。

1. **入口**：学生在 ChatWindow 提问，前端走 SSE（`/api/chat`）或 WebSocket（`/api/run/chat`）。
   - `api/chat.py` / `api/run.py`
2. **鉴权 + 多租户隔离**：JWT 解析用户 → `check_course_access(course_id, user_id)` 校验"该学生是否选了这门课"，结果 Redis 缓存。未选课直接 403（"自由问答" general 课程除外，任何登录用户可用）。
   - `api/courses.py` 的 `check_course_access`
3. **编排选路**：`TurnRuntimeManager.start_turn()` → `CourseOrchestrator.handle(context)` 按 `context.mode`（chat / deep_solve / deep_research / quiz / summarize / vision）选出对应 Capability。
   - `services/session/turn_runtime.py`、`core/orchestrator.py`、`core/registry.py`
4. **Agent Loop**：chat 走 `ChatPipeline`（组装 system prompt → `run_agent_loop`）；解题/研究/出题各自是独立多阶段 pipeline，每阶段一次 `run_agent_loop`。进入 tool_calls 多轮循环：
   - 每轮 LLM 决定调哪些工具（rag 检索 / web_search / ask_user / solve_*）→ 并行分发（最多 8 个并发）→ 把 `role=tool` 结果塞回对话 → 下一轮。
   - 最后一轮强制 `tools=None`，LLM 必须输出文字答案。
   - `core/capabilities/chat_pipeline.py`、`core/agentic/loop.py`
5. **流式回传**：每一步通过 `StreamBus` 发事件（thinking / tool_call / tool_result / token / answer / done），前端逐个渲染。
   - **最终答案轮是真流式**：chunk-by-chunk 透传（见取舍 2）。
   - `core/stream_bus.py`、`core/agentic/loop.py`（`live_sink`）

---

## 4. 五个亮点（每个准备讲 3 分钟 + 回答追问）

### 亮点 1：四大能力对齐 DeepTutor（tool_calls 版）

> "DeepTutor 用首行 label（FINISH/TOOL/THINK）做流程门控，我对比后选了 tool_calls，用代码编排 + 工具状态机实现等价能力深度。"

- **单层 Capability + 共享 `run_agent_loop`**：chat / deep_solve / deep_research / quiz / summarize / vision 各是一个 `BaseCapability` 薄壳，注册到 `CapabilityRegistry`。
  - `core/capability_protocol.py`、`core/registry.py`
- **等价门控靠代码而非 label**：solve 用 `SolveSession` 状态机（`solve_plan`/`finish_step`/`replan` 工具）；quiz 三阶段编排（explore→plan→quiz）；research 用 `DynamicTopicQueue` + `CitationManager`。
  - `core/solve/session.py`、`core/question/pipeline.py`、`core/research/pipeline.py`
- **价值**：加新模式只写 capability + 工具，核心 `run_agent_loop` 不动；流程刚性由代码保证，不依赖弱模型遵守 label。

### 亮点 2：provider 无关的 Agent Loop

> "我的一套 ReAct 循环，能跑 GPT、Claude、通义、DeepSeek，不用为每家改逻辑。"

- Loop 只认 `client.chat.completions.create()`（OpenAI 协议）。
- **AnthropicAdapter** 把 Claude 原生的 `tool_use` 流式协议**桥接成 OpenAI chunk 格式**——delta.content / delta.tool_calls 字段对齐。
  - `core/llm/providers/anthropic_adapter.py`
- 14 个 provider 注册表，工厂按 backend 选 client。
  - `core/llm/provider_registry.py`、`core/llm/provider_factory.py`

### 亮点 3：多租户全链路隔离（SaaS 核心，对标 DeepTutor 的关键差异）

> "DeepTutor 是单机的，admin 只是部署管理员。我做的是真多租户——从数据库到运行时到向量库全链路隔离。"

- **DB 层**：`knowledge_bases.owner_id`（KB 属于哪个教师）+ `enrollments`（学生-课程选课关系，唯一约束）。
  - `alembic/versions/002_roles_invites_enrollments.py`
- **运行时**：`check_course_access` 在每个 chat / WS 入口强制校验，Redis 缓存结果。
- **向量库**：LightRAG 按 `course_id` 分独立 workspace（`lightrag_store/course_*`），物理隔离。
- **角色**：student / teacher / admin 三级 RBAC，教师凭 invite_code 注册升级。

### 亮点 4：可靠性工程（面向真实流量）

> "不是 demo 级的'调通就行'，我做了生产级的容错。"

- **熔断器**（CLOSED/OPEN/HALF_OPEN）+ **指数退避重试** + **fallback 兜底 client**。
  - `core/llm/reliability.py`
- **进程级 LLM 并发信号量**（`MAX_CONCURRENT_LLM`），防瞬时打爆上游。
- **RAG 检索缓存 + 权限缓存**（Redis），热路径降延迟。
- **健康检查**带 LLM/Redis/DB 探针 + 熔断器重置。
  - `main.py` 的 `/api/health`

### 亮点 5：流式 + 交互暂停 + 真流式优化

> "AI 能边想边说，还能中途反问学生。"

- **StreamBus**：per-turn 事件总线，支持历史回放（断线重连不丢事件）、stage 上下文。
  - `core/stream_bus.py`
- **ask_user 工具**：loop 可**暂停**等学生回答卡片，把回答写回 `role=tool` 消息继续推理。WS 入口双向，SSE 入口优雅降级。
  - `core/agentic/loop.py`（`wait_for_user_reply`）
- **真流式**：最终答案轮 chunk-by-chunk 透传，首字延迟 ≈ 首 token 时间（见取舍 2）。

---

## 5. 难点与取舍（面试加分项，证明"懂为什么"）

### 取舍 1：tool_calls vs label-driven loop（对标 DeepTutor）

> "我对比过两种 agent loop 范式，有意识地选了 tool_calls。"

| | tool_calls（我的选择） | label-driven（DeepTutor） |
|---|---|---|
| 机制 | OpenAI 原生 function calling | 第一行文本标签 THINK/TOOL/FINISH |
| 鲁棒性 | provider 保证格式 | 依赖 prompt 工程，弱模型可能不遵守 |
| 主流度 | 2024 后主流（LangGraph / OpenAI Agents SDK 同路） | ReAct 系传统，provider 无关 |
| 控制语义 | 只有"调工具" | 丰富（PAUSE/APPEND/REPLAN…） |
| 流式 | 最终轮需额外处理 | 第一行知答案，原生友好 |

**我选 tool_calls 的理由**：①鲁棒性由 provider 保证；②符合主流；③配 AnthropicAdapter 已能跨 provider。
**代价**：最终答案轮流式弱于 label——我用"最后一轮 live 透传"补齐（见取舍 2）。

### 取舍 2：真流式 vs 假 token

> "旧实现是假流式——LLM 生成完整答案后按 8 字符切片假装逐字发。我改成了真流式。"

- **问题**：旧 `_emit_as_tokens` 把完整字符串切块，首字延迟 = 整段生成时间。
- **根因**：tool_calls loop 必须等整轮收完才能判断"有没有 tool_calls"，所以单轮内难边收边发。
- **我的方案**：利用 loop 已有的"最后一轮 `tools=None` 强制收尾"——这轮 LLM 物理上不可能输出 tool_calls，可放心 chunk-by-chunk 透传。给 `_one_round` 加 `live_sink`，最后一轮传入。
  - `core/agentic/loop.py`：`_one_round(..., live_sink)` + `is_final_round`
- **覆盖**：RAG/解题/研究"调几轮工具 → 最后一轮总结"的主力长答案场景（80% 收益）。
- **未做（已知优化项）**：非最后一轮的"提前收尾"乐观流式，涉及前端语义，留后续。

### 取舍 3：从旧多-agent 重构到统一 loop + 工具状态机

> "旧版 solve/research 各有一套 planner/solver/writer 显式多 agent 通信，重复且难维护。我重构成'对同一个 run_agent_loop 的多次调用'，流程刚性用工具状态机保证。"

- 旧的 `_legacy/` 多 agent 编排已归档删除。
- quiz / research：对 `run_agent_loop` 的多次调用（每阶段一次），`dataclasses.replace` 隔离上下文。
- solve：单次 `run_agent_loop`，靠 `SolveSession` 状态机 + `solve_plan`/`finish_step`/`replan` 工具驱动流程。
- **价值**：消除重复循环逻辑，加新模式只写 capability。

### 取舍 4（已知局限，主动说显得坦诚）

- **多 worker 下 turn 状态**：`TurnRuntimeManager` 是进程内单例，Gunicorn 4 worker 下 WS 重连可能落到别的 worker。生产需要粘性路由或换 Redis 共享状态。
- **chat 旧路径**：`/api/chat/lightrag`（旧意图分类链路）仍在兼容，但安全护栏已统一到 `ChatPipeline`，两条路径共用同一套 `safety_pipeline`。新功能一律走 `chat_mode` / `/api/run/{cap}`。

---

## 6. SaaS 差异化（对比开源单机 DeepTutor）

| 能力 | 单机 DeepTutor | 本项目（SaaS） |
|---|---|---|
| 多租户 / 角色 | 无 | admin/teacher/student RBAC |
| 教师后台 | 无 | 建课、KB CRUD、6 个学情看板、邀请码 |
| 课程码自助入课 | 无 | join_code + enrollments 隔离 |
| 学情分析 | 无 | 活跃度趋势、高频问题、学生问答回放、知识图谱 |
| 异步任务队列 | 同步 | ARQ worker（索引/解题/研究/定时总结） |
| 用量统计 | 无 | （规划中）token 按用户/课程计量 |

**讲法**："DeepTutor 明确声明不做付费产品，它的多用户只是部署特性。我的护城河是真正的多租户 SaaS + 教师端 + 学情闭环——这是它能讲出而 DeepTutor 仓库讲不出的差异。"

---

## 7. 简历话术

### 一段话（中文）
> 课程 AI SaaS（对标 DeepTutor 的多租户升级版）：FastAPI + ARQ + PostgreSQL 后端，React 19 前端。设计单层 Capability + tool_calls 驱动 Agent Loop（四大能力 chat/solve/research/quiz 对齐 DeepTutor，跨 14 个 LLM provider，含 Claude 协议适配），双引擎 RAG（LightRAG 图谱 + LlamaIndex），SSE/WebSocket 双通道真流式。实现多租户 RBAC、教师学情看板、异步任务队列、熔断器+fallback 可靠性工程。建立中文课程 QA 评测集量化 faithfulness/citation recall。

### One-liner（英文）
> Built a multi-tenant AI tutoring SaaS (DeepTutor-inspired): FastAPI + ARQ backend with a tool-calling agent loop spanning 14 LLM providers, dual-engine RAG (LightRAG + LlamaIndex), real-time streaming, RBAC multi-tenancy, teacher analytics dashboards, and circuit-breaker reliability engineering.

---

## 8. 可能被追问的问题（准备答案）

- **Q: 为什么不用 LangChain/LangGraph？** → A: 自研 loop 是为了 provider 无关 + 控制流完全可控；LangGraph 是 graph 抽象，我的 capability 已够用且更轻。
- **Q: Agent Loop 怎么防止无限循环？** → A: max_iterations 预算，最后一轮强制 `tools=None` 收尾。
- **Q: 流式断线怎么办？** → A: StreamBus 历史回放 + WS reconnect 事件。
- **Q: 多租户怎么保证数据不串？** → A: 三层隔离（DB owner_id/enrollment + 运行时 check_course_access + LightRAG 独立 workspace）。
- **Q: 怎么衡量 RAG 质量？** → A: 自建评测集，LLM-as-judge 量化 faithfulness / citation recall（见 eval/）。
