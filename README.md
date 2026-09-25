# 课程学习 Agent

面向学生的智能课程学习助手。支持多课程管理、知识库构建、流式对话、深度研究、智能出题等功能。

## 功能

- **多课程 / 多知识库**：管理员/教师创建课程，上传文档建立知识库；学生用课程码自助入课
- **RAG 对话**：基于 LightRAG 图谱检索 + LlamaIndex 向量检索，支持知识溯源
- **四大能力**（统一通过 `chat_mode` 或 WS `/api/run/{cap}` 触发）：
  - **Chat**：tool_calls 驱动的 Agent Loop，安全护栏 + RAG/web 工具
  - **Deep Solve**：单 loop + `solve_plan/finish_step/replan` 工具 + `SolveSession` 状态机
  - **Deep Research**：rephrase → decompose → research(队列+并行) → reporting(带引用)
  - **Quiz**：explore → plan → quiz，6 类题型 + JSON schema 校验
- **流式响应**：SSE + WebSocket 双通道推送 thinking / tool_call / token / answer 等事件（真流式）
- **会话持久化**：PostgreSQL 存储多会话历史，支持对话模式切换

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | Python / FastAPI / Uvicorn / Gunicorn |
| 数据库 | PostgreSQL 16 + SQLAlchemy (asyncpg) |
| 缓存 / 队列 | Redis 7 (API 缓存 + ARQ 任务队列) |
| RAG 引擎 | LightRAG (图谱) + LlamaIndex (向量) |
| LLM | 通义千问 / OpenAI 兼容接口 (DashScope) |
| 认证 | JWT (PyJWT) + bcrypt |
| 限流 | slowapi (Redis 后端) |
| 监控 | Prometheus + prometheus-fastapi-instrumentator |
| 前端 | React 19 / TypeScript / Vite / TailwindCSS |
| 容器化 | Docker / Docker Compose |

## 架构

```
┌───────────────────────────────────────────────────────┐
│  入口                                                  │
│  POST /api/chat?chat_mode=...  (SSE 流式)             │
│  WS   /api/run/{capability}     (统一 WS)             │
│  Bot  QQ / 飞书                 (共享同一引擎)         │
└────────────────────────┬──────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────┐
│  TurnRuntimeManager  →  CourseOrchestrator            │
│  (按 mode 选 Capability，StreamBus fan-out 事件)       │
│   ChatCapability / DeepSolve / DeepResearch / Quiz     │
│   → run_agent_loop (tool_calls 多轮，真流式)           │
└──────┬────────────────────────────────┬───────────────┘
       │ 工具面                          │ 双引擎 RAG
┌──────▼──────────────┐    ┌────────────▼──────────────┐
│ rag / web_search    │    │ LightRAG(图谱+向量)        │
│ ask_user / solve_*  │    │ LlamaIndex(多模态文档)     │
└─────────────────────┘    └───────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│  PostgreSQL（用户/会话/KB）│ Redis（缓存/队列）        │
│  ARQ Worker（索引/解题/研究/定时总结）               │
└─────────────────────────────────────────────────────┘
```

事件经 `StreamBus` 统一 fan-out 给 SSE / WebSocket 消费者（含断线回放）。详见 [`backend/docs/ARCHITECTURE.md`](backend/docs/ARCHITECTURE.md)。

## 快速开始

### 1. 配置环境变量（两份 .env，缺一不可）

```bash
# ① 项目根目录：docker-compose 变量（数据库密码等）
cp .env.example .env
# 编辑 .env，至少修改 POSTGRES_PASSWORD（生产环境必改）

# ② backend/：应用层配置（LLM Key、JWT 密钥等）
cd backend
cp .env.example .env
# 编辑 .env，至少填入：
#   LLM__API_KEY=sk-xxx            # LLM 主模型 Key
#   SECURITY__JWT_SECRET=<随机长字符串>
```

> 注意：配置采用嵌套命名（`组__变量`，双下划线分隔），如 `LLM__API_KEY`、`SECURITY__JWT_SECRET`、`DB__URL`。旧的扁平名（`DASHSCOPE_API_KEY`、`JWT_SECRET`、`DATABASE_URL`）已不再生效。

### 2. Docker Compose 一键启动

```bash
docker compose up -d
```

包含服务：`postgres`、`redis`、`backend`（4 workers）、`worker`（ARQ 后台任务）、`frontend`（Nginx 反代 :80）、`mcp-gateway`（MCP 工具 sidecar）、`sandbox`（编程题判分沙箱）

> ⚠️ **升级已部署环境必须带 `--build`**：`docker compose up -d --build`。
> backend/worker 以源码卷热生效，但**前端静态产物烤在镜像里**——不带 `--build`
> 会出现后端新、前端旧且无任何报错（详见 docs/DELIVERY.md 六节）。

访问：http://localhost

> 评测容器是独立文件，按需叠加启动（非生产运行时）：
> `docker compose -f docker-compose.yml -f docker-compose.eval.yml up -d`

### 3. 本地开发（不用 Docker）

**后端**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
# 需要本地 PostgreSQL 和 Redis；连接串在 backend/.env 里改：
#   DB__URL=postgresql+asyncpg://postgres:<密码>@localhost:5432/course_agent
#   DB__REDIS_URL=redis://localhost:6379/0
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

**ARQ Worker（另开终端）**
```bash
cd backend
python -m arq worker.WorkerSettings
```

**前端**
```bash
cd frontend
npm install
npm run dev
```

访问：http://localhost:5173

### 4. 运行测试

**后端集成测试**（使用 SQLite in-memory，不需要外部服务）
```bash
cd backend
TESTING=1 pytest tests/ -v
```

**前端单测**
```bash
cd frontend
npm test
```

## 项目结构

```
backend/
├── main.py                  # FastAPI 入口、lifespan、全局异常处理
├── worker.py                # ARQ Worker：索引/解题/研究/定时汇总
├── settings/                # 环境变量统一配置（base.py，嵌套组__变量）
├── api/
│   ├── auth.py              # /api/auth  注册/登录/JWT
│   ├── chat.py              # /api/chat  SSE 流式对话（chat_mode 选能力）
│   ├── run.py               # WS /api/run/{capability}  统一 WS 入口
│   ├── courses.py           # /api/courses  课程列表 + 入课
│   ├── sessions.py          # /api/sessions  会话 CRUD
│   ├── upload.py            # /api/upload  文件上传（鉴权）
│   ├── admin.py             # /api/admin  知识库管理（仅管理员）
│   ├── teacher.py           # /api/teacher  课程 CRUD + 学情分析（教师）
│   ├── kb_indexing.py       # 索引触发/进度/主题图查询（admin/teacher 共享）
│   ├── question.py / question_notebook.py  # 出题 + 答题本
│   ├── lightrag.py          # /api/lightrag  LightRAG 查询接口
│   ├── memory.py            # /api/memory  用户记忆管理
│   ├── mcp.py / user_llm.py / llm.py / search_config.py / skill_knowledge.py / skills.py
│   └── bot.py               # IM Bot webhook（QQ / 飞书）
├── core/
│   ├── orchestrator.py      # CourseOrchestrator：按 mode 选 Capability
│   ├── registry.py          # CapabilityRegistry（chat/solve/research/quiz）
│   ├── capability_protocol.py  # BaseCapability 协议
│   ├── context.py           # UnifiedContext（单轮上下文）
│   ├── stream_bus.py        # StreamBus 事件总线（fan-out + 回放）
│   ├── pipeline_common.py   # 公共上下文层（记忆/学情门控/限权检索）
│   ├── prompt_loader.py     # YAML 提示词加载（四能力共用）
│   ├── agentic/
│   │   ├── loop.py          # run_agent_loop：tool_calls 调度内核
│   │   └── tool_dispatch.py # 并行工具执行（≤8 并发）
│   ├── capabilities/        # chat_pipeline + 各 capability 薄壳
│   ├── solve/               # deep_solve：pipeline + SolveSession + 工具
│   ├── research/            # deep_research：pipeline + 动态队列 + CitationManager
│   ├── question/            # quiz：explore → plan → quiz pipeline
│   ├── agent/               # tool_registry（rag/web_search/ask_user/solve_*）
│   ├── memory/              # 三层记忆栈（L2 摘要/L3 mem0/巩固/掌握度）
│   ├── db/
│   │   ├── database.py      # SQLAlchemy 模型 + 异步引擎
│   │   ├── cache.py         # Redis 缓存工具（FAQ / 课程 / 权限）
│   │   └── limiter.py       # slowapi 限流器（测试环境可禁用）
│   ├── llm/
│   │   ├── llm.py           # AsyncOpenAI 客户端封装 + 多 provider 注册表
│   │   └── reliability.py   # 熔断器 + 重试 + fallback
│   └── rag/
│       ├── ingestion.py     # 文档摄入管线
│       ├── registry.py      # 检索后端注册表（lightrag / llamaindex_pg）
│       ├── chunking/        # 可插拔切块策略
│       ├── parsing/         # 文档解析（Docling / MinerU）
│       ├── llamaindex/      # LlamaIndex PG 向量后端
│       └── lightrag/        # LightRAG 图谱后端
├── services/session/        # TurnRuntimeManager（单回合生命周期）
├── tests/                   # 1400+ 集成/单元测试（SQLite in-memory）

frontend/src/
├── App.tsx
├── components/
│   ├── pages/LoginPage.tsx  # 登录/注册页
│   └── ...                  # 聊天、课程、侧边栏等组件
├── services/
│   ├── api.ts               # REST / SSE 接口调用
│   └── auth.ts              # JWT 存取、登录/注册
├── types/index.ts           # TypeScript 类型定义
└── __tests__/               # Vitest 单元测试
```

## 关键环境变量

均通过 `组__变量` 嵌套命名注入（详见 `backend/.env.example` 内完整清单）：

| 变量 | 说明 | 默认 |
|------|------|------|
| `LLM__API_KEY` | LLM 主模型 API Key | 必填 |
| `SECURITY__JWT_SECRET` | JWT 签名密钥（生产必须修改，默认值会 fail-fast） | dev-secret-... |
| `SECURITY__ALLOWED_ORIGINS` | CORS 白名单（生产不允许 `*`） | * |
| `DB__URL` | PostgreSQL 连接串 | 本机默认 |
| `DB__REDIS_URL` | Redis 连接串 | redis://localhost:6379/0 |
| `ENVIRONMENT` | `development` / `production` | development |
| `BACKEND_WORKERS` | Gunicorn worker 数量 | 4 |
| `DB__POOL_SIZE` | 数据库连接池大小 | 10 |
| `DB__MAX_OVERFLOW` | 连接池溢出上限 | 15 |
| `LLM_TIMEOUT_SEC` | LLM 调用超时（秒） | 120 |

根目录 `.env`（docker-compose 读取）：`POSTGRES_PASSWORD`、`ENVIRONMENT` 等，见 `.env.example`。

## 健康检查

`GET /api/health` 返回 DB、Redis、LLM 三项状态：

```json
{
  "status": "ok",
  "checks": {
    "db": "ok",
    "redis": "ok",
    "llm": "ok (api_key configured)"
  }
}
```

任意项异常时 HTTP 状态码变为 `503`。
