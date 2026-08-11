# 检索模式选择器改造：后端优先 + LightRAG 子模式

- 日期：2026-08-06
- 范围：聊天界面的「检索模式」选择器（前端）+ rag 路由（后端小改）
- 不在范围：建库表单、数据库表结构、Alembic 迁移、建库/索引流程

## 1. 背景与现状

聊天界面已存在一个检索模式选择器（`frontend/src/components/chat/ChatWindow.tsx`）：

- 状态 `ragMode: 'auto'|'mix'|'naive'|'local'`，默认 `'auto'`（`:237`）
- 渲染两处：移动端 `:1533-1545`、桌面端 `:1621-1633`，显隐条件 `useKb && ragEnabled && hasLightrag`
- 选项：自动(auto) / 混合(mix) / 向量(naive) / 实体(local)
- 随 `/api/chat` 体字段 `rag_mode` 发送（`frontend/src/services/api.ts:192`）
- 只建 pgvector 的课程：选择器隐藏，强制发 `auto`

后端路由在 `backend/core/agent/tool_registry.py:_execute_rag`（`:48-130`），路由梯子（`:66-100`）：

1. `mode ∈ {mix,naive,local,global}` 且 lightrag 就绪 → LightRAG 该模式（最优先）
2. `auto` + relationship 策略 + lightrag 就绪 → LightRAG 图增强
3. 否则 pgvector 就绪 → pgvector ← **pgvector 仅在此被动命中**
4. 否则 lightrag 就绪 → LightRAG naive
5. 否则 → 「未就绪」

`rag_mode` 白名单在 `backend/api/chat.py:64`：`{"auto","mix","naive","local","global"}`，非法回退 `auto`。`rag_mode` 经 `chat.py → context.py → loop dispatch → tool_registry call_kwargs["mode"]` 透传。

`GET /api/courses` 已返回 `index_backends: string[]`（就绪后端列表，`backend/api/courses.py:33`），前端据此决定选择器显隐。

上个 commit `a34d40f` 移除了**建库表单**里的索引后端下拉框，理由：路由由「已就绪的 builds」决定，与创建时填的字段无关。本设计**不改建库表单**，理由仍然成立。

## 2. 问题（为什么必须动后端）

用户要在 UI 上**显式选 pgvector**，并去掉「自动」。但当前**没有任何 `rag_mode` 值能强制走 pgvector**：选 mix/naive/local 永远走 LightRAG；pgvector 只能在 `auto` 下被间接触发。删掉 auto 等于删掉去 pgvector 的唯一入口 → UI 上的 pgvector 选项会变成摆设。

## 3. 目标

- 选择器拆两级：顶层选后端（lightrag / pgvector，**只列 `index_backends` 里已就绪的**）；选 lightrag 时再出子框 向量/实体/混合；选 pgvector 无子框。
- 去掉「自动」选项。
- 后端支持「强制 pgvector」。
- 默认（两后端都建好）：**pgvector**（决策已定，B）。
- 零表结构改动、零迁移。

## 4. 设计

### 4.1 前端（`ChatWindow.tsx`）

**状态拆分**（替换单一 `ragMode`）：

```ts
const hasLightrag = indexBackends.includes('lightrag')
const hasPg = indexBackends.includes('llamaindex_pg')
const [ragBackend, setRagBackend] = useState<'lightrag' | 'llamaindex_pg'>('llamaindex_pg')
const [ragLightragMode, setRagLightragMode] = useState<'mix' | 'naive' | 'local'>('mix')
```

**默认值随就绪后端重算**（`useEffect` 依赖 `indexBackends`）：
- 两者都有 → `llamaindex_pg`（B）
- 仅 pg → `llamaindex_pg`
- 仅 lightrag → `lightrag`

**派生要发送的值**：

```ts
const ragModeToSend = ragBackend === 'llamaindex_pg' ? 'llamaindex_pg' : ragLightragMode
```

**UI 显隐**（替换 `:1533` 与 `:1621` 两处；整体仍受 `useKb && ragEnabled && hasLightrag` 门控——即 lightrag 已建才显示选择器，与现状一致；只建 pg 时选择器依旧隐藏）：
- 顶层后端下拉：**仅当 lightrag 与 pg 同时就绪时显示**（只有一种后端时省略顶层，无意义）。
- 子框 向量(naive)/实体(local)/混合(mix)：仅当 `ragBackend === 'lightrag'` 时显示。
- 去掉「自动」option。

**避免重复**：现有两处选择器已是近乎复制；改成两级（条件顶层 + 条件子框）后重复加倍。抽一个内部小组件 `<RagModeSelector>`（props：`backend, lightragMode, onBackend, onLightragMode, showTop`），桌面/移动两处复用，逻辑单点维护。

**发送处改写**（`:1102`）：`useKb ? ragModeToSend : 'mix'`（KB 关闭时 rag 工具未启用，值无意义，沿用 `'mix'` 占位以最小改动）。`ragModeToSend` 是派生值，需替换 `useCallback` 依赖数组中原 `ragMode`（`:1171`）为 `ragModeToSend`（或其两个源 `ragBackend, ragLightragMode`）。`api.ts` 入参已是 `string`，默认值/兜底不动（`ragModeToSend` 恒非空，`|| 'auto'` 不触发）。

### 4.2 后端

**`backend/api/chat.py:64`** 白名单加 `llamaindex_pg`：

```python
if rag_mode not in {"auto", "mix", "naive", "local", "global", "llamaindex_pg"}:
    rag_mode = "auto"
```

**`backend/core/agent/tool_registry.py:_execute_rag`** 在 LightRAG 模式分支**之前**插入最高优先级分支：

```python
if mode == "llamaindex_pg":
    # 用户显式选 pgvector：强制走 pg 向量，要求 pg 已就绪（不看 strategy）
    if "llamaindex_pg" not in ready:
        return ToolResult(content="（该课程未构建 pgvector 索引。）", success=False)
    retriever = get_retriever("llamaindex_pg")
    content = await retriever.retrieve_context(course_id=course_id, query=query, top_k=top_k)
elif mode in ("mix", "naive", "local", "global"):
    ...  # 原逻辑不动
```

`auto` 路由梯子（`:81-100`）**原样保留**——UI 不再发 auto，但留作兜底（其他非 chat 入口可能依赖，不审计、不动）。同步更新 `_execute_rag` docstring（`:50-57`）与 `ChatRequest.rag_mode` 描述（`:45`）。

### 4.3 选择 → `rag_mode` 映射

| UI 选择 | 发送 `rag_mode` | 后端路由 |
|---|---|---|
| pgvector | `llamaindex_pg` | 新分支 → 强制 pgvector |
| lightrag → 向量 | `naive` | LightRAG naive |
| lightrag → 实体 | `local` | LightRAG local |
| lightrag → 混合 | `mix` | LightRAG mix |

## 5. 调研依据

- **模式词汇**：LightRAG 原生 `QueryParam(mode=...)` 取 `naive/local/global/hybrid/mix`。向量/实体/混合 ↔ `naive/local/mix` 的映射**与现有 UI 完全一致**（`ChatWindow.tsx:1540-1543` 已是 naive=向量、local=实体、mix=混合），非新发明。pgvector 检索器（`LlamaIndexRetriever`）**无 mode 参数**（恒为 dense+稀疏 RRF），故选 pg 时无子框，符合用户描述。
- **信号传递选型（为什么 A 不选 B）**：A=往 `rag_mode` 塞新哨兵值 `llamaindex_pg`，1 白名单 + 1 分支，复用现成透传链；B=新增独立字段 `rag_backend`，语义更干净但要改 `ChatRequest/context.py/loop dispatch/tool_registry` 四处签名，侵入大。一个下拉框不值得，选 A。
- **「绑定已建索引」**：`index_backends` 已由 `GET /api/courses` 返回（源自 `kb_builds` 就绪行），前端只读，零表改动。
- 本项非研究型决策（无 PDF/embedding/向量库选型），不涉及 arxiv；属标准「级联下拉 + 路由哨兵」工程模式。

## 6. 验证

- **后端单测**：
  - `_execute_rag(mode="llamaindex_pg")` 在 pg 就绪时调用 `LlamaIndexRetriever.retrieve_context`（mock `get_retriever` 断言被调后端）；pg 未就绪时返回 `success=False` 的提示。
  - `chat.py` 白名单：`rag_mode="llamaindex_pg"` 原样通过，不被回退 `auto`。
  - 回归：`auto` 路由梯子行为不变（既有 rag 路由测试应全绿）。
- **前端**：`tsc` 通过；手动验证三种就绪组合（仅 lightrag / 仅 pg / 两者）的选择器显隐与默认值。
- **并发/安全**：本改动不涉及共享态或鉴权，无需 interleaving/攻击者视角推演。

## 7. 改动清单

前端：
- `frontend/src/components/chat/ChatWindow.tsx` — 状态拆分 + 默认 effect + 两处选择器 UI 改两级 + 发送处改 `ragModeToSend`
- `frontend/src/services/api.ts` — `streamChat` 的 `ragMode` 入参类型注释（仅文档性，签名已是 `string`）

后端：
- `backend/api/chat.py` — 白名单加 `llamaindex_pg` + 字段描述
- `backend/core/agent/tool_registry.py` — `_execute_rag` 新增 pgvector 强制分支 + docstring

测试：
- `backend/tests/` — 新增/扩展 rag 路由测试覆盖 `llamaindex_pg`

文档：
- `backend/docs/ARCHITECTURE.md` — 检索路由段落补「显式 pgvector」一句（如已有检索路由描述）
