import type { Course, Session, SSEEvent, Message as AppMessage, RagChunk, ChatMode, AttachmentInfo } from '../types'
import { authHeaders } from './auth'

async function readErrorMessage(res: Response): Promise<string> {
  const text = await res.text()
  if (!text) return `HTTP ${res.status}`
  try {
    const data = JSON.parse(text) as { detail?: string; message?: string }
    return data.detail || data.message || text
  } catch {
    return text
  }
}

function checkUnauthorized(res: Response) {
  if (res.status === 401) {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    window.location.reload()
  }
}

// ---------------------------------------------------------------------------
// Courses
// ---------------------------------------------------------------------------

export async function fetchCourses(): Promise<Course[]> {
  let res: Response
  try {
    res = await fetch('/api/courses', { headers: authHeaders() })
  } catch {
    throw new Error('无法连接后端服务，请确认后端已启动')
  }
  if (!res.ok) throw new Error(await readErrorMessage(res))

  const data = await res.json()
  return Array.isArray(data.courses) ? data.courses : []
}

export async function joinCourseByCode(
  joinCode: string,
): Promise<{ course_id: string; name: string; already_enrolled: boolean }> {
  const res = await fetch('/api/courses/join', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ join_code: joinCode }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// ---------------------------------------------------------------------------
// Sessions (auth-protected)
// ---------------------------------------------------------------------------

export async function fetchSessions(courseId?: string): Promise<Session[]> {
  const params = courseId ? `?course_id=${courseId}` : ''
  const res = await fetch(`/api/sessions${params}`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.sessions || []
}

export async function createSession(courseId: string, title?: string, mode: ChatMode = 'chat'): Promise<Session> {
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ course_id: courseId, title: title || '新对话', mode }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function updateSessionMode(sessionId: string, mode: ChatMode): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/mode`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ mode }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
}

export async function fetchMessages(sessionId: string): Promise<AppMessage[]> {
  const res = await fetch(`/api/sessions/${sessionId}/messages`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.messages || []
}

export async function saveMessage(
  sessionId: string,
  role: string,
  content: string,
  msgType: string = 'text',
  metadata?: Record<string, unknown>,
): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ role, content, msg_type: msgType, metadata }),
  })
  checkUnauthorized(res)
}

// ---------------------------------------------------------------------------
// Upload (auth-protected)
// ---------------------------------------------------------------------------

export async function uploadImage(file: File): Promise<{ filename: string; path: string }> {
  const form = new FormData()
  form.append('file', file)
  let res: Response
  try {
    res = await fetch('/api/upload', { method: 'POST', body: form, headers: authHeaders() })
  } catch {
    throw new Error('上传失败：无法连接后端服务')
  }
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// ---------------------------------------------------------------------------
// Chat stream (auth-protected, new SSE protocol)
// ---------------------------------------------------------------------------

export async function chatStream(
  courseId: string,
  message: string,
  history: { role: string; content: string }[],
  imagePath?: string,
  sessionId?: string,
  chatMode: ChatMode = 'chat',
  signal?: AbortSignal,
  onEvent?: (event: SSEEvent) => void,
  onError?: (err: string) => void,
  enabledTools: string[] = [],
  modelProfileId?: string,
  attachments?: AttachmentInfo[],
  ragMode: string = 'mix',
): Promise<{ aborted: boolean }> {
  const isAbortError = (err: unknown) => {
    if (err instanceof DOMException) return err.name === 'AbortError'
    if (err instanceof Error) return err.name === 'AbortError'
    return false
  }
  // 统一走新路径 /api/chat（run_agent_loop + rag tool）；ragEnabled 只决定 tools 是否含 rag，
  // 不再切旧 /api/chat/lightrag（deprecated，tech-decisions#4 迁移完成）
  const endpoint = '/api/chat'
  const traceId =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(16).slice(2, 10)
  const t0 = performance.now()
  const devTrace = Boolean((import.meta as { env?: { DEV?: boolean } }).env?.DEV)
  const logTrace = (stage: string, extra?: Record<string, unknown>) => {
    if (!devTrace) return
    const elapsedMs = Math.round(performance.now() - t0)
    console.log(`[lightrag-trace=${traceId}] ${stage} t=${elapsedMs}ms`, extra || {})
  }

  let res: Response
  try {
    logTrace('send', { endpoint })
    res = await fetch(endpoint, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', 'X-Trace-Id': traceId, ...authHeaders() },
      body: JSON.stringify({
        course_id: courseId,
        message,
        history,
        image_path: imagePath || null,
        session_id: sessionId || null,
        chat_mode: chatMode,
        tools: enabledTools,
        model_profile_id: modelProfileId || null,
        attachments: attachments ?? [],
        rag_mode: ragMode || 'mix',
      }),
    })
  } catch (err) {
    if (isAbortError(err) || signal?.aborted) return { aborted: true }
    onError?.('无法连接后端服务，请确认后端已启动')
    return { aborted: false }
  }

  if (res.status === 401) {
    checkUnauthorized(res)
    return { aborted: false }
  }

  logTrace('response_headers', { status: res.status })
  if (!res.ok || !res.body) {
    onError?.(await readErrorMessage(res))
    return { aborted: false }
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let firstEventLogged = false
  let firstTokenLogged = false
  let aborted = false

  try {
    while (true) {
      if (signal?.aborted) {
        aborted = true
        break
      }
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() || ''
      let shouldStop = false

      for (const eventBlock of events) {
        const jsonStr = eventBlock
          .split('\n')
          .map((line) => line.trimEnd())
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n')
        if (!jsonStr) continue

        try {
          const event = JSON.parse(jsonStr) as SSEEvent
          if (!firstEventLogged) {
            logTrace('first_event', { type: event.type })
            firstEventLogged = true
          }
          if (event.type === 'tool_result' && !event.chunks) {
            const rawContexts = (event as SSEEvent & { contexts?: unknown[] }).contexts
            if (Array.isArray(rawContexts) && rawContexts.length > 0) {
              const chunks: RagChunk[] = rawContexts.map((ctx, idx) => {
                if (typeof ctx === 'string') {
                  return { content: ctx, source: 'lightrag', score: 1 - idx * 0.01 }
                }
                if (ctx && typeof ctx === 'object') {
                  const row = ctx as Record<string, unknown>
                  const content = String(row.content ?? row.text ?? row.chunk ?? '')
                  const source = String(row.source ?? row.file ?? 'lightrag')
                  const scoreRaw = Number(row.score)
                  return {
                    content,
                    source,
                    score: Number.isFinite(scoreRaw) ? scoreRaw : 1 - idx * 0.01,
                  }
                }
                return { content: String(ctx), source: 'lightrag', score: 1 - idx * 0.01 }
              }).filter((c) => c.content.trim().length > 0)

              if (chunks.length > 0) {
                event.chunks = chunks
              }
            }
          }
          onEvent?.(event)
          if (event.type === 'token' && !firstTokenLogged) {
            logTrace('first_token')
            firstTokenLogged = true
          }
          if (event.type === 'done' || event.type === 'error') {
            logTrace('stream_end', { type: event.type })
            shouldStop = true
            break
          }
        } catch {
          // skip malformed JSON
        }
      }

      if (shouldStop) {
        try {
          await reader.cancel()
        } catch {
          // ignore reader cancellation errors
        }
        break
      }
    }
  } catch (err) {
    if (isAbortError(err) || signal?.aborted) {
      aborted = true
    } else {
      onError?.('流式连接中断，请重试')
    }
  }

  if (aborted) {
    logTrace('stream_aborted')
    try {
      await reader.cancel()
    } catch {
      // ignore reader cancellation errors
    }
  }
  return { aborted }
}

/**
 * 触发"立即回答"：让正在思考的 turn 在下一轮顶部基于已有信息直接作答。
 * 流式过程中前端"立即回答"按钮调用（fire-and-forget，不中断当前 SSE 流）。
 * 返回是否成功触发（turn 存在且未结束）；404/网络异常返回 false，前端静默处理。
 */
export async function requestAnswerNow(turnId: string): Promise<boolean> {
  try {
    const res = await fetch('/api/chat/answer_now', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ turn_id: turnId }),
    })
    checkUnauthorized(res)
    return res.ok
  } catch {
    return false
  }
}




// ---------------------------------------------------------------------------
// Memory Graph / Dashboard
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string
  type: string
  label: string
  risk?: number
  mastery?: number
  importance?: number
  severity?: number
  error_count?: number
  status?: string
  notes?: string
  examples?: string[]
  related_points?: string[]
  correction_suggestions?: string[]
  updated_at?: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: { source: string; target: string; relation: string }[]
}

export interface GraphsResponse {
  knowledge_graph: GraphData
  error_graph: GraphData
}

export async function fetchGraphs(): Promise<GraphsResponse> {
  const res = await fetch('/api/memory/graph', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function deleteGraphNode(nodeId: string): Promise<GraphsResponse> {
  const res = await fetch('/api/memory/graph/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ node_id: nodeId }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export interface DashboardData {
  summary: string
  profile: string
  high_risk_points: GraphNode[]
  frequent_errors: GraphNode[]
  knowledge_node_count: number
  error_node_count: number
}

export async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch('/api/memory/dashboard', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// ---------------------------------------------------------------------------
// Output Skills
// ---------------------------------------------------------------------------

export interface OutputSkill {
  id: string
  title: string
  description: string
  instruction: string
  enabled: boolean
  course_id: string
}

export async function fetchSkills(): Promise<OutputSkill[]> {
  const res = await fetch('/api/skills', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skills || []
}

export async function createSkill(skill: { title: string; description: string; instruction: string; course_id?: string }): Promise<OutputSkill> {
  const res = await fetch('/api/skills', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(skill),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skill
}

export async function toggleSkill(skillId: string, enabled: boolean): Promise<OutputSkill> {
  const res = await fetch(`/api/skills/${skillId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ enabled }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skill
}

export async function deleteSkill(skillId: string): Promise<void> {
  const res = await fetch(`/api/skills/${skillId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

// ---------------------------------------------------------------------------
// Skill Knowledge（DeepTutor 式 SKILL.md 知识包，区别于上面的 OutputSkill 补充框）
// ---------------------------------------------------------------------------

export interface SkillSummaryEntry {
  name: string
  description: string
  available: boolean
  missing: string[]
  always: boolean
  source: 'personal' | 'course' | 'builtin'
}

export interface SkillKnowledgeDetail {
  name: string
  description: string
  always: boolean
  content: string
  source: 'personal' | 'course' | 'builtin'
  read_only: boolean
}

export async function fetchSkillKnowledge(courseId?: string): Promise<SkillSummaryEntry[]> {
  const params = courseId ? `?course_id=${courseId}` : ''
  const res = await fetch(`/api/skill-knowledge${params}`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skills || []
}

export async function fetchSkillDetail(name: string, courseId?: string): Promise<SkillKnowledgeDetail> {
  const params = courseId ? `?course_id=${courseId}` : ''
  const res = await fetch(`/api/skill-knowledge/${encodeURIComponent(name)}${params}`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function createSkillKnowledge(payload: {
  name: string; description: string; content: string; always?: boolean; course_id?: string
}): Promise<SkillSummaryEntry> {
  const res = await fetch('/api/skill-knowledge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skill
}

export async function updateSkillKnowledge(name: string, payload: {
  description?: string; content?: string; always?: boolean; rename_to?: string; course_id?: string
}): Promise<SkillSummaryEntry> {
  const res = await fetch(`/api/skill-knowledge/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.skill
}

export async function deleteSkillKnowledge(name: string, courseId?: string): Promise<void> {
  const params = courseId ? `?course_id=${courseId}` : ''
  const res = await fetch(`/api/skill-knowledge/${encodeURIComponent(name)}${params}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

// ---------------------------------------------------------------------------
// MCP server 配置（admin only，部署级）
// ---------------------------------------------------------------------------

export interface McpServerStatus {
  name: string
  transport: string
  status: string // connecting | connected | error | disabled
  error: string
  tools: { name: string; description: string }[]
}

export interface McpServerConfig {
  type?: string | null
  command: string
  args: string[]
  env: Record<string, string>
  cwd: string
  url: string
  headers: Record<string, string>
  tool_timeout: number
  enabled_tools: string[]
  enabled: boolean
}

export interface McpServersResponse {
  servers: McpServerStatus[]
  config: Record<string, McpServerConfig>
}

export interface McpProbeResult {
  ok: boolean
  tools: { name: string; description: string }[]
  error: string
}

export async function fetchMcpServers(): Promise<McpServersResponse> {
  const res = await fetch('/api/mcp/servers', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function upsertMcpServer(name: string, config: McpServerConfig): Promise<void> {
  const res = await fetch(`/api/mcp/servers/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(config),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function deleteMcpServer(name: string): Promise<void> {
  const res = await fetch(`/api/mcp/servers/${encodeURIComponent(name)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function testMcpProbe(config: McpServerConfig): Promise<McpProbeResult> {
  const res = await fetch('/api/mcp/probe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(config),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// --- MCP 个人启用开关（非 admin 用户：只读目录 + 启用勾选；server 进程系统级共享）---

export interface McpCatalogServer {
  name: string
  transport: string
  enabled_globally: boolean
  connected: boolean
  tools: { name: string; description: string }[]
}

export async function fetchMcpCatalog(): Promise<McpCatalogServer[]> {
  const res = await fetch('/api/mcp/servers/catalog', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.servers || []
}

export async function fetchMyMcpEnabled(): Promise<string[]> {
  const res = await fetch('/api/mcp/me/enabled', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.enabled_servers || []
}

export async function setMyMcpEnabled(enabledServers: string[]): Promise<string[]> {
  const res = await fetch('/api/mcp/me/enabled', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ enabled_servers: enabledServers }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.enabled_servers || []
}

// ---------------------------------------------------------------------------
// 题目笔记本（错题收藏/分类）
// ---------------------------------------------------------------------------

export interface NotebookCategory {
  id: number
  name: string
  created_at: number
  entry_count: number
}

export interface NotebookEntry {
  id: number
  session_id: string
  session_title: string
  question_id: string
  question: string
  question_type: string
  options: Record<string, string>
  correct_answer: string
  explanation: string
  difficulty: string
  user_answer: string
  is_correct: boolean
  bookmarked: boolean
  created_at: number
  updated_at: number
  categories?: NotebookCategory[] | null
}

export async function fetchNotebookEntries(params: {
  category_id?: number
  bookmarked?: boolean
  is_correct?: boolean
  limit?: number
  offset?: number
} = {}): Promise<{ items: NotebookEntry[]; total: number }> {
  const qs = new URLSearchParams()
  if (params.category_id != null) qs.set('category_id', String(params.category_id))
  if (params.bookmarked != null) qs.set('bookmarked', String(params.bookmarked))
  if (params.is_correct != null) qs.set('is_correct', String(params.is_correct))
  if (params.limit != null) qs.set('limit', String(params.limit))
  if (params.offset != null) qs.set('offset', String(params.offset))
  const res = await fetch(`/api/question/notebook/entries?${qs}`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function updateNotebookEntry(
  id: number,
  payload: { bookmarked?: boolean; followup_session_id?: string },
): Promise<void> {
  const res = await fetch(`/api/question/notebook/entries/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function deleteNotebookEntry(id: number): Promise<void> {
  const res = await fetch(`/api/question/notebook/entries/${id}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function fetchNotebookCategories(): Promise<NotebookCategory[]> {
  const res = await fetch('/api/question/notebook/categories', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function createNotebookCategory(name: string): Promise<NotebookCategory> {
  const res = await fetch('/api/question/notebook/categories', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ name }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function deleteNotebookCategory(id: number): Promise<void> {
  const res = await fetch(`/api/question/notebook/categories/${id}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

// ---------------------------------------------------------------------------
// TutorBot 管理（IM bot，QQ/飞书/Web 渠道）
// ---------------------------------------------------------------------------

export interface BotInstance {
  bot_id: string
  owner_id?: string
  name: string
  description: string
  persona: string
  running: boolean
  channels: Record<string, unknown>
  course_id?: string
}

export async function fetchBots(): Promise<BotInstance[]> {
  const res = await fetch('/api/bot/list', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.bots || []
}

export async function createBot(payload: {
  bot_id: string
  name: string
  description?: string
  persona?: string
  course_id?: string
  channels?: Record<string, Record<string, unknown>>
}): Promise<BotInstance> {
  const res = await fetch('/api/bot/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.bot
}

export async function startBot(botId: string): Promise<void> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/start`, {
    method: 'POST',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function stopBot(botId: string): Promise<void> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/stop`, {
    method: 'POST',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function deleteBot(botId: string): Promise<void> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function updateBot(
  botId: string,
  payload: {
    name?: string
    description?: string
    persona?: string
    course_id?: string
  },
): Promise<void> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function sendBotMessage(botId: string, content: string, chatId = 'web'): Promise<string> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ content, chat_id: chatId }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.response || ''
}

export async function fetchBotHistory(
  botId: string,
  limit = 100,
): Promise<{ role: string; content: string; timestamp?: string }[]> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/history?limit=${limit}`, {
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.messages || []
}

// --- IM 账号绑定（打通 web↔QQ/飞书 user_id，长期记忆跨渠道）---

export interface SocialBinding {
  id: string
  platform: string
  platform_user_id: string
  chat_id?: string
  display_name?: string
  created_at?: number
}

export async function generateBindCode(): Promise<{ code: string; expires_in: number }> {
  const res = await fetch('/api/bot/bind/code', { method: 'POST', headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function fetchMyBindings(): Promise<SocialBinding[]> {
  const res = await fetch('/api/bot/bindings/me', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.bindings || []
}

export async function deleteBinding(bindingId: string): Promise<void> {
  const res = await fetch(`/api/bot/bind/${encodeURIComponent(bindingId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

// ---------------------------------------------------------------------------
// Bot Reminders（定时通知） + Notifications（web 触达）
// ---------------------------------------------------------------------------

export interface BotReminderSchedule {
  kind: string
  at_ms: number | null
  every_seconds: number | null
  expr: string | null
  tz: string | null
}

export interface BotReminder {
  id: string
  name: string
  message: string
  schedule: BotReminderSchedule
  channel: string
  chat_id: string
  enabled: boolean
  created_at_ms: number
  state: {
    next_run_at_ms: number | null
    last_run_at_ms: number | null
    last_status: string | null
    last_error: string | null
  }
}

export async function listReminders(botId: string): Promise<BotReminder[]> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/reminders`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.reminders || []
}

export async function createReminder(botId: string, payload: {
  name?: string
  message: string
  channel?: string
  chat_id?: string
  session_key?: string
  schedule: BotReminderSchedule
}): Promise<BotReminder> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/reminders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.reminder
}

export async function deleteReminder(botId: string, jobId: string): Promise<void> {
  const res = await fetch(`/api/bot/${encodeURIComponent(botId)}/reminders/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export interface BotNotificationItem {
  id: string
  bot_id: string
  content: string
  read: boolean
  created_at: number
}

export async function fetchNotifications(unread = false): Promise<{ notifications: BotNotificationItem[]; unread_count: number }> {
  const res = await fetch(`/api/bot/notifications${unread ? '?unread=true' : ''}`, { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function markNotificationRead(id: string): Promise<void> {
  const res = await fetch(`/api/bot/notifications/${encodeURIComponent(id)}/read`, {
    method: 'POST',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

// ---------------------------------------------------------------------------
// LLM 供应商 profile（对标 DeepTutor：admin 预配 provider 池，用户对话时下拉临时切换）
// ---------------------------------------------------------------------------

export interface LlmProviderSpec {
  name: string
  backend: string
  default_api_base: string
  env_key: string
}

export interface LlmProfileSelectable {
  id: string
  name: string
  binding: string
  text_model: string
  fast_model: string
  vision_model: string
  base_url_configured: boolean
  api_key_configured: boolean
  active: boolean
}

export interface LlmProfileAdmin {
  id: string
  name: string
  binding: string
  api_key: string
  base_url: string
  api_version: string
  text_model: string
  fast_model: string
  vision_model: string
  embedding_model: string
  embedding_api_key: string
  embedding_base_url: string
  fallback_api_key: string
  fallback_base_url: string
  fallback_model: string
  active: boolean
}

export interface LlmProfilePayload {
  name?: string
  binding?: string
  api_key?: string
  base_url?: string
  api_version?: string
  text_model?: string
  fast_model?: string
  vision_model?: string
  embedding_model?: string
  embedding_api_key?: string
  embedding_base_url?: string
  fallback_api_key?: string
  fallback_base_url?: string
  fallback_model?: string
}

export interface LlmProbeResult {
  ok: boolean
  binding?: string | null
  model?: string
  error?: string
  // 用户级 provider 测试扩展（/llm/me/test 同时探测对话 + 视觉两路）；
  // admin 的 /probe 不返回这两个字段 → undefined，前端据此区分是否展示视觉栏。
  text?: { ok: boolean; model: string; error: string }
  vision?: { ok: boolean; model: string; error: string; warning?: string | null } | null
}

export async function fetchLlmProviders(): Promise<LlmProviderSpec[]> {
  const res = await fetch('/api/llm/providers', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.providers || []
}

export async function fetchLlmProfilesSelectable(): Promise<{ profiles: LlmProfileSelectable[]; active: string }> {
  const res = await fetch('/api/llm/profiles/selectable', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function fetchLlmProfilesAdmin(): Promise<{ profiles: LlmProfileAdmin[]; active: string }> {
  const res = await fetch('/api/llm/profiles', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function upsertLlmProfile(profileId: string, payload: LlmProfilePayload): Promise<LlmProfileAdmin> {
  const res = await fetch(`/api/llm/profiles/${encodeURIComponent(profileId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.profile
}

export async function deleteLlmProfile(profileId: string): Promise<void> {
  const res = await fetch(`/api/llm/profiles/${encodeURIComponent(profileId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function setActiveLlmProfile(profileId: string): Promise<void> {
  const res = await fetch('/api/llm/active', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ profile_id: profileId }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function testLlmProfile(profileId: string): Promise<LlmProbeResult> {
  const res = await fetch(`/api/llm/profiles/${encodeURIComponent(profileId)}/test`, {
    method: 'POST',
    headers: authHeaders(),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function probeLlmProfile(payload: LlmProfilePayload): Promise<LlmProbeResult> {
  const res = await fetch('/api/llm/probe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// ── 用户级 LLM provider（/llm/me，多租户隔离；覆盖平台默认）────────────────
// 学生可自配 binding/key/text/fast/vision 模型；embedding 平台统一不开放（per-course
// 共享库要求 embedding 一致）。vision 模型用于「主回答模型不支持视觉」时的两阶段图片描述。
export interface UserProviderView {
  // 对话供应商
  binding: string
  api_key_set: boolean
  base_url: string
  api_version: string
  text_model: string
  // 视觉独立供应商
  vision_binding: string
  vision_api_key_set: boolean
  vision_base_url: string
  vision_model: string
}

export interface UserProviderPayload {
  // 对话供应商
  binding: string
  api_key: string
  base_url: string
  api_version: string
  text_model: string
  // 视觉独立供应商（可异于对话供应商：对话走 deepseek，视觉走 dashscope/qwen-vl）
  vision_binding: string
  vision_api_key: string
  vision_base_url: string
  vision_model: string
}

export async function fetchMyLlmProvider(): Promise<UserProviderView> {
  const res = await fetch('/api/llm/me', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function upsertMyLlmProvider(payload: UserProviderPayload): Promise<unknown> {
  const res = await fetch('/api/llm/me', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function deleteMyLlmProvider(): Promise<void> {
  const res = await fetch('/api/llm/me', { method: 'DELETE', headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function testMyLlmProvider(payload: UserProviderPayload): Promise<LlmProbeResult> {
  const res = await fetch('/api/llm/me/test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

// ── 联网搜索配置（admin 默认 + 用户覆盖）─────────────────────────────────
export interface SearchProviderInfo {
  id: string
  name: string
  description: string
  supports_answer: boolean
  requires_api_key: boolean
}

export interface SearchConfigPayload {
  provider: string
  api_key: string
  base_url: string
  max_results: number
  proxy: string
}

export interface SearchProbeResult {
  ok: boolean
  provider: string
  error: string
}

export async function fetchSearchProviders(): Promise<SearchProviderInfo[]> {
  const res = await fetch('/api/search_config/providers', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.providers || []
}

export async function fetchSearchAdminConfig(): Promise<SearchConfigPayload> {
  const res = await fetch('/api/search_config/admin', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function putSearchAdminConfig(payload: SearchConfigPayload): Promise<SearchConfigPayload> {
  const res = await fetch('/api/search_config/admin', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.config
}

export async function fetchMySearchConfig(): Promise<SearchConfigPayload & { has_override: boolean }> {
  const res = await fetch('/api/search_config/me', { headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

export async function putMySearchConfig(payload: SearchConfigPayload): Promise<SearchConfigPayload> {
  const res = await fetch('/api/search_config/me', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  const data = await res.json()
  return data.config
}

export async function deleteMySearchConfig(): Promise<void> {
  const res = await fetch('/api/search_config/me', { method: 'DELETE', headers: authHeaders() })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
}

export async function probeSearchConfig(
  payload: { provider: string; api_key: string; base_url: string },
): Promise<SearchProbeResult> {
  const res = await fetch('/api/search_config/probe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
}

