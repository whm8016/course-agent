import type { Course, Session, SSEEvent, Message as AppMessage } from '../types'
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

export async function createSession(courseId: string, title?: string): Promise<Session> {
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ course_id: courseId, title: title || '新对话' }),
  })
  checkUnauthorized(res)
  if (!res.ok) throw new Error(await readErrorMessage(res))
  return res.json()
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
  onEvent?: (event: SSEEvent) => void,
  onError?: (err: string) => void,
) {
  let res: Response
  try {
    res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({
        course_id: courseId,
        message,
        history,
        image_path: imagePath || null,
        session_id: sessionId || null,
      }),
    })
  } catch {
    onError?.('无法连接后端服务，请确认后端已启动')
    return
  }

  if (res.status === 401) {
    checkUnauthorized(res)
    return
  }

  if (!res.ok || !res.body) {
    onError?.(await readErrorMessage(res))
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const jsonStr = line.slice(6).trim()
      if (!jsonStr) continue

      try {
        const event = JSON.parse(jsonStr) as SSEEvent
        onEvent?.(event)
      } catch {
        // skip malformed JSON
      }
    }
  }
}
