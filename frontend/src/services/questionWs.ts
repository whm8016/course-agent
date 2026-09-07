/** WebSocket 出题 & 深度研究服务 */
import { getToken } from './auth'
import type { AttachmentInfo } from '../types'

export interface QuestionRequirement {
  knowledge_point: string
  preference?: string
  difficulty?: string
  question_type?: string
}

export interface QuestionGeneratePayload {
  /** 统一 WS /api/run/quiz 协议（首条消息默认 start_turn） */
  course_id: string
  question: string
  metadata?: {
    count?: number
    requirement?: string
    [key: string]: unknown
  }
  language?: string
  attachments?: AttachmentInfo[]
  /** 显式启用的工具。出题默认 ['rag']：始终基于该课程知识库检索，避免模型凭空发挥 */
  tools?: string[]
  /** 同会话此前的对话（OpenAI message 格式）：切出题前刚聊的内容，后端以"背景摘录"形态带入 */
  history?: { role: string; content: string }[]
  /** 所属会话 id：后端据此加载 L2 摘要（长会话更早历史的压缩） */
  session_id?: string
}

export interface QuestionFollowupPayload {
  question_context: Record<string, unknown>
  history_context: string
  user_message: string
  language?: string
}

export interface QuestionGenMessage {
  type: string
  [key: string]: unknown
}

/** WS 连接句柄：close 主动关闭；send 向后端发消息（如 ask_user 的 submit_user_reply）。 */
export interface WsHandle {
  close: () => void
  send: (msg: object) => void
}

function wsBaseUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

function withToken(url: string): string {
  const token = getToken()
  return token ? `${url}?token=${encodeURIComponent(token)}` : url
}

// ============================================================
// Deep Research
// ============================================================

/** 与 backend ResearchSource 一致 */
export type DeepResearchSource = 'kb' | 'web' | 'papers'

export interface DeepResearchPayload {
  /** 统一 WS /api/run/deep_research 协议（首条消息默认 start_turn） */
  course_id: string
  question: string
  metadata?: {
    sources?: DeepResearchSource[]
  }
  language?: string
  attachments?: AttachmentInfo[]
  /** 同会话此前的对话（OpenAI message 格式）：切研究前刚聊的内容，后端以"背景摘录"形态带入 */
  history?: { role: string; content: string }[]
  /** 所属会话 id：后端据此加载 L2 摘要（长会话更早历史的压缩） */
  session_id?: string
}

export interface DeepResearchMessage {
  type: string
  [key: string]: unknown
}

export interface DeepResearchHandlers {
  onOpen?: () => void
  onMessage?: (msg: DeepResearchMessage) => void
  onClose?: () => void
  onError?: () => void
}

function deepResearchWsUrl(): string {
  return withToken(`${wsBaseUrl()}/api/run/deep_research`)
}

export function connectDeepResearch(
  payload: DeepResearchPayload,
  handlers: DeepResearchHandlers = {},
): WsHandle {
  const ws = new WebSocket(deepResearchWsUrl())

  const close = () => {
    try {
      // 主动关闭：先解绑回调再 close——浏览器对主动关闭同样触发 onclose/onerror，
      // 不解绑会被当"意外断开"（research 的 onClose 会带 resume_research_id 自动
      // 重连，把刚取消/停止的大纲确认卡原样弹回）
      ws.onclose = null
      ws.onerror = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    } catch {
      // ignore
    }
  }

  const send = (msg: object) => {
    try {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg))
    } catch {
      // ignore
    }
  }

  ws.onopen = () => {
    handlers.onOpen?.()
    ws.send(JSON.stringify(payload))
  }

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data)) as DeepResearchMessage
      handlers.onMessage?.(msg)
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return { close, send }
}

function questionGenerateWsUrl(): string {
  return withToken(`${wsBaseUrl()}/api/run/quiz`)
}

function questionFollowupWsUrl(): string {
  return withToken(`${wsBaseUrl()}/api/question/followup`)
}

export interface QuestionGenHandlers {
  onMessage?: (msg: QuestionGenMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: () => void
  /** Fired when followup stream ends successfully (`type: done`). */
  onComplete?: () => void
}

/**
 * 建立 WebSocket 连接后立刻发送 payload，返回 close 函数。
 */
export function connectQuestionGenerate(
  payload: QuestionGeneratePayload,
  handlers: QuestionGenHandlers = {},
): WsHandle {
  const ws = new WebSocket(questionGenerateWsUrl())

  const close = () => {
    try {
      // 主动关闭：先解绑回调再 close（同 connectDeepResearch，防误报"意外关闭"/重连）
      ws.onclose = null
      ws.onerror = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    } catch {
      // ignore
    }
  }

  const send = (msg: object) => {
    try {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg))
    } catch {
      // ignore
    }
  }

  ws.onopen = () => {
    handlers.onOpen?.()
    ws.send(JSON.stringify(payload))
  }

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data)) as QuestionGenMessage
      handlers.onMessage?.(msg)
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return { close, send }
}

/** 单题追问（/api/question/followup），服务端流式 token + answer + done */
export function connectQuestionFollowup(
  payload: QuestionFollowupPayload,
  handlers: QuestionGenHandlers = {},
): () => void {
  const ws = new WebSocket(questionFollowupWsUrl())

  const close = () => {
    try {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    } catch {
      // ignore
    }
  }

  ws.onopen = () => {
    handlers.onOpen?.()
    ws.send(JSON.stringify(payload))
  }

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data)) as QuestionGenMessage
      handlers.onMessage?.(msg)
      if (msg.type === 'done') {
        handlers.onComplete?.()
      }
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return close
}
