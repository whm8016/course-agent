/** WebSocket 出题 & 深度研究服务 */
import { getToken } from './auth'
import type { AgentMetadata, AttachmentInfo } from '../types'

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
  /** result 事件携带聚合 usage 等元信息（与 chat 的 done.metadata 同源，后端 usage_payload） */
  metadata?: AgentMetadata
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

/** WS 句柄工厂（三个 connect 共用）。close 先解绑 onclose/onerror 再关——浏览器对主动
 * 关闭同样派发 close 事件，不解绑会被上层当"意外断开"（research 的 onClose 带
 * resume_research_id 自动重连，会把刚取消/停止的大纲确认卡原样弹回）。 */
function makeWsHandle(ws: WebSocket): WsHandle {
  const close = () => {
    try {
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
  return { close, send }
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
  /** result 事件携带聚合 usage 等元信息（与 chat 的 done.metadata 同源，后端 usage_payload） */
  metadata?: AgentMetadata
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
  const { close, send } = makeWsHandle(ws)

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
 *
 * 断线恢复（后端 /api/run/quiz 协议）：收到 `session` 记 turn_id、按 bus 事件计数
 * 维护 seq；意外断开（非主动 close、未 done/error）在宽限窗内带
 * `subscribe_turn{turn_id, after_seq}` 重连续流，已流式生成的题目不丢、整轮不重付。
 * 指数退避最多 3 次，耗尽发 `ws_reconnect_failed` 消息后按普通 onClose 收尾。
 */
export function connectQuestionGenerate(
  payload: QuestionGeneratePayload,
  handlers: QuestionGenHandlers = {},
): WsHandle {
  const MAX_RETRY = 3

  let turnId: string | null = null
  let seq = 0 // 已收到的 bus 事件数（session 为直发不走 bus，不计入——after_seq 口径）
  let attempt = 0
  let settled = false // 主动 close 或已收 done/error：之后的 close 事件按正常收尾，不重连
  let active: WsHandle | null = null
  let retryTimer: number | undefined

  const connect = (first: boolean) => {
    const ws = new WebSocket(questionGenerateWsUrl())
    active = makeWsHandle(ws)

    ws.onopen = () => {
      handlers.onOpen?.()
      ws.send(
        JSON.stringify(
          first
            ? payload
            : { type: 'subscribe_turn', turn_id: turnId, after_seq: seq },
        ),
      )
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data)) as QuestionGenMessage
        if (msg.type === 'session' && typeof msg.turn_id === 'string') {
          turnId = msg.turn_id
        } else {
          seq += 1
        }
        attempt = 0 // 有事件到达即连接健康，后续再闪断从头计数
        if (msg.type === 'done' || msg.type === 'error') settled = true
        handlers.onMessage?.(msg)
      } catch {
        handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
      }
    }

    // 初始建连阶段（还没有 turn）的 error 才上抛——有 turn 后错误由重连逻辑接管，
    // 逐次尝试的瞬时错误不该直接砸出「连接失败」
    ws.onerror = () => {
      if (!turnId) handlers.onError?.()
    }

    ws.onclose = () => {
      if (settled) {
        handlers.onClose?.()
        return
      }
      if (turnId && attempt < MAX_RETRY) {
        attempt += 1
        retryTimer = window.setTimeout(() => connect(false), 1000 * 2 ** (attempt - 1))
        return
      }
      if (turnId) {
        // 有 turn 但重连耗尽：显式通知上层（区别于从未连上的初始失败）
        handlers.onMessage?.({ type: 'ws_reconnect_failed', attempts: attempt })
      }
      handlers.onClose?.()
    }
  }

  connect(true)

  return {
    close: () => {
      settled = true
      if (retryTimer !== undefined) window.clearTimeout(retryTimer)
      active?.close()
    },
    send: (msg: object) => active?.send(msg),
  }
}

/** 单题追问（/api/question/followup），服务端流式 token + answer + done */
export function connectQuestionFollowup(
  payload: QuestionFollowupPayload,
  handlers: QuestionGenHandlers = {},
): () => void {
  const ws = new WebSocket(questionFollowupWsUrl())
  const { close } = makeWsHandle(ws)

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
