# 课程学习 Agent

面向学生的智能课程学习助手，支持"邮票赏析"和"电路分析"两门课程。基于 **LangGraph 多 Agent 编排** 和 **ChromaDB 向量检索** 构建。

## 功能

- **多 Agent 编排**：LangGraph 状态图驱动的 Router → 教学/出题/总结/视觉 四大 Agent
- **RAG 知识检索**：ChromaDB 持久化向量库 + DashScope Embedding，支持知识溯源
- **智能出题**：Agent 自动从知识库生成选择题，前端交互式作答
- **图片分析**：邮票上传赏析（多模态视觉模型）
- **Agent 可观测性**：前端实时展示思考过程、工具调用、知识来源
- **会话持久化**：SQLite 存储对话历史，支持多会话管理
- **流式响应**：SSE 推送 thinking / tool_call / tool_result / quiz / answer 等丰富事件

## 技术栈

- **后端**: Python / FastAPI / LangGraph / LangChain / ChromaDB / SQLite
- **前端**: React / TypeScript / Vite / TailwindCSS
- **LLM**: 通义千问 (Qwen API via DashScope)

## 架构

```
用户消息 → Router Agent（意图分类）
              ├── teach   → search_knowledge → RAG 增强回答
              ├── quiz    → search_knowledge → 生成测验题
              ├── summarize → 对话历史总结
              └── vision  → analyze_image → 视觉模型分析
```

## 快速开始

### 1. 配置 API Key

```bash
cd backend
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 启动后端

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开浏览器访问 http://localhost:5173

## 项目结构

```
backend/
├── main.py                 # FastAPI 入口
├── config.py               # 配置（模型、路径、参数）
├── api/
│   ├── chat.py             # /api/chat SSE 流式对话
│   ├── courses.py          # /api/courses 课程列表
│   ├── upload.py           # /api/upload 图片上传
│   └── sessions.py         # /api/sessions 会话 CRUD
├── core/
│   ├── orchestrator.py     # LangGraph 多 Agent 状态图
│   ├── tools.py            # Tool 定义（search_knowledge, generate_quiz, analyze_image）
│   ├── rag.py              # ChromaDB 向量检索
│   ├── llm.py              # LLM 调用封装
│   ├── prompts.py          # 系统提示词（课程/路由/出题/总结）
│   └── memory.py           # SQLite 会话存储
└── knowledge/              # 课程知识库（Markdown）

frontend/src/
├── App.tsx
├── components/
│   ├── ChatWindow.tsx      # 主聊天窗口
│   ├── MessageBubble.tsx   # 消息气泡（支持多类型渲染）
│   ├── ThinkingProcess.tsx # Agent 思考过程面板
│   ├── SourceCard.tsx      # 知识溯源卡片
│   ├── QuizCard.tsx        # 交互式测验卡片
│   ├── SessionList.tsx     # 会话历史列表
│   ├── Sidebar.tsx         # 侧边栏
│   ├── CourseSelector.tsx  # 课程选择
│   └── ImageUpload.tsx     # 图片上传
├── services/api.ts         # API 调用 + SSE 事件解析
└── types/index.ts          # TypeScript 类型定义
```
