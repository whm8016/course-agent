# 课程学习 Agent

面向学生的智能课程学习助手。支持多课程管理、知识库构建、流式对话、深度研究、智能出题等功能。

## 功能

- **多课程 / 多知识库**：管理员/教师创建课程，上传文档建立知识库；学生用课程码自助入课
- **RAG 对话**：基于 LightRAG 图谱检索 + LlamaIndex 向量检索，支持知识溯源
- **Deep Research**：多阶段自动研究 Pipeline（Planning → Searching → Reporting），进度实时推送
- **Deep Solve**：Plan → ReAct → Write 三阶段解题，支持 RAG 工具调用
- **智能出题**：从知识库自动生成题目，支持仿题与答题交互
- **流式响应**：SSE + WebSocket 双通道推送 thinking / trace / progress / result 等事件
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
                    ┌─────────────────────────────────────────┐
                    │            Nginx (反向代理)               │
                    └──────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               │     FastAPI (4 workers)        │
               │  /api/chat   SSE流式对话       │
               │  /api/deep-research  WS        │
               │  /api/deep-solve     WS        │
               │  /api/question       WS        │
               │  /api/admin  /api/teacher      │
               └──────┬────────────┬────────────┘
                      │            │
          ┌───────────┘     ┌──────┘
          │                 │
    ┌─────▼─────┐    ┌──────▼──────┐
    │PostgreSQL  │    │   Redis      │
    │(会话/用户  │    │(缓存/限流/   │
    │ /知识库)   │    │ ARQ 任务队列)│
    └───────────┘    └──────┬──────┘
                            │ RPUSH job:*:events
                     ┌──────▼──────┐
                     │ ARQ Worker  │
                     │(索引/研究/  │
                     │  解题任务)  │
                     └─────────────┘
```

### 长任务执行路径

```
客户端 WS ──► FastAPI enqueue_job ──► Redis 任务队列
                     │
                     └─► WS 轮询 LRANGE job:{id}:events
                                          ▲
                                          │ RPUSH 进度事件
                              ARQ Worker 运行 Pipeline
```

## 快速开始

### 1. 配置 API Key

```bash
cd backend
cp .env.example .env
# 编辑 .env，至少填入：
#   DASHSCOPE_API_KEY=sk-xxx
#   JWT_SECRET=<随机长字符串>
```

### 2. Docker Compose 一键启动

```bash
docker compose up -d
```

包含服务：`postgres`、`redis`、`backend`（4 workers）、`worker`（ARQ 后台任务）、`frontend`（Nginx）

访问：http://localhost

### 3. 本地开发（不用 Docker）

**后端**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
# 需要本地 PostgreSQL 和 Redis，或修改 DATABASE_URL / REDIS_URL
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
├── worker.py                # ARQ Worker：索引/研究/解题任务
├── config.py                # 环境变量统一配置
├── api/
│   ├── auth.py              # /api/auth  注册/登录/JWT
│   ├── chat.py              # /api/chat  SSE 流式对话（LightRAG / 普通）
│   ├── courses.py           # /api/courses  课程列表 + 入课
│   ├── sessions.py          # /api/sessions  会话 CRUD
│   ├── upload.py            # /api/upload  图片上传（鉴权）
│   ├── admin.py             # /api/admin  知识库管理（仅管理员）
│   ├── teacher.py           # /api/teacher  课程 CRUD + 索引（教师）
│   ├── llama_rag.py         # /api/admin/kb/../llamaindex  向量索引
│   ├── deep_research.py     # /api/deep-research/run  WS
│   ├── deep_solve.py        # /api/deep-solve/run  WS
│   ├── question.py          # /api/question  出题 WS
│   ├── lightrag.py          # /api/lightrag  LightRAG 查询接口
│   ├── memory.py            # /api/memory  用户记忆管理
│   └── sse.py               # /api/sse  SSE 示例
├── core/
│   ├── arq_pool.py          # ARQ 连接池单例
│   ├── db/
│   │   ├── database.py      # SQLAlchemy 模型 + 异步引擎
│   │   ├── cache.py         # Redis 缓存工具（FAQ / 课程 / 权限）
│   │   └── limiter.py       # slowapi 限流器（测试环境可禁用）
│   ├── llm/
│   │   ├── llm.py           # AsyncOpenAI 客户端封装
│   │   └── prompts.py       # 系统提示词管理
│   └── rag/
│       ├── lightrag_engine.py       # LightRAG 检索引擎
│       ├── ingestion.py             # 文档摄入（LlamaIndex → LightRAG）
│       └── llamaindex/              # LlamaIndex Pipeline
├── tests/
│   ├── conftest.py          # pytest fixtures（SQLite / httpx / 认证头）
│   ├── test_auth.py         # 注册/登录/鉴权测试
│   ├── test_sessions.py     # 会话 CRUD 测试
│   ├── test_admin.py        # 管理员权限测试
│   ├── test_upload.py       # 文件上传/访问鉴权测试
│   ├── test_chat.py         # 对话接口测试
│   └── test_websocket.py    # WS 鉴权测试

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
    ├── setup.ts
    ├── auth.test.ts         # auth service 单测
    ├── api.test.ts          # api service 单测
    └── LoginPage.test.tsx   # 组件渲染/交互测试
```

## 关键环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 通义千问 API Key | 必填 |
| `JWT_SECRET` | JWT 签名密钥（生产必须修改） | dev-secret-... |
| `DATABASE_URL` | PostgreSQL 连接串 | sqlite+aiosqlite://... |
| `REDIS_URL` | Redis 连接串 | redis://localhost:6379/0 |
| `ENVIRONMENT` | `development` / `production` | development |
| `BACKEND_WORKERS` | Gunicorn worker 数量 | 4 |
| `DB_POOL_SIZE` | 数据库连接池大小 | 10 |
| `DB_MAX_OVERFLOW` | 连接池溢出上限 | 20 |
| `LLM_TIMEOUT_SEC` | LLM 调用超时（秒） | 120 |

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
