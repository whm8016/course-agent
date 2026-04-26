import type { Course, Session, SSEEvent, Message as AppMessage, RagChunk, ChatMode } from '../types'
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
    res = await fetch('/api/courses')
  } catch {
    throw new Error('无法连接后端服务，请确认后端已启动')
  }
  if (!res.ok) throw new Error(await readErrorMessage(res))

  const data = await res.json()
  return Array.isArray(data.courses) ? data.courses : []
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
  ragEnabled: boolean = false,
  enabledTools: string[] = [],
): Promise<{ aborted: boolean }> {
  const isAbortError = (err: unknown) => {
    if (err instanceof DOMException) return err.name === 'AbortError'
    if (err instanceof Error) return err.name === 'AbortError'
    return false
  }
  const endpoint = ragEnabled ? '/api/chat/lightrag' : '/api/chat'
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




//练习用SSE

export type SseTokenHandler = (word: string) => void
export type SseDoneHandler = () => void
export type SseErrorHandler = (message: string) => void
export async function streamSseDemo(
  text: string,
  onToken: (word: string) => void,
  onDone?: () => void,
) {
  const res = await fetch('/api/sse/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok || !res.body) {
    throw new Error(await res.text())
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() || ''

    for (const block of blocks) {
      let eventName = 'message'
      const dataLines: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
      }
      const data = dataLines.join('\n')
      if (eventName === 'token') onToken(data)
      if (eventName === 'done') onDone?.()
    }
  }
}