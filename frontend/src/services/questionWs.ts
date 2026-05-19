/** WebSocket 出题 & 深度研究服务 */

export interface QuestionRequirement {
  knowledge_point: string
  preference?: string
  difficulty?: string
  question_type?: string
}

export interface QuestionGeneratePayload {
  kb_name: string
  count: number
  language?: string
  requirement: QuestionRequirement
}

export interface QuestionMimicPayload {
  mode: 'upload' | 'parsed'
  kb_name: string
  max_questions?: number
  language?: string
  pdf_data?: string
  pdf_name?: string
  paper_path?: string
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

function wsBaseUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

// ============================================================
// Deep Research
// ============================================================

/** 与 backend/core/research/request_config.py ResearchMode 一致 */
export type DeepResearchMode = 'notes' | 'report' | 'comparison' | 'learning_path'

/** 与 backend ResearchDepth 一致 */
export type DeepResearchDepth = 'quick' | 'standard' | 'deep' | 'manual'

/** 与 backend ResearchSource 一致 */
export type DeepResearchSource = 'kb' | 'web' | 'papers'

export interface DeepResearchPayload {
  type: 'start'
  topic: string
  config?: {
    mode?: DeepResearchMode
    depth?: DeepResearchDepth
    sources?: DeepResearchSource[]
  }
  kb_name?: string
  research_id?: string
  language?: string
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
  return `${wsBaseUrl()}/api/deep-research/run`
}

export function connectDeepResearch(
  payload: DeepResearchPayload,
  handlers: DeepResearchHandlers = {},
): () => void {
  const ws = new WebSocket(deepResearchWsUrl())

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
      const msg = JSON.parse(String(ev.data)) as DeepResearchMessage
      handlers.onMessage?.(msg)
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return close
}

function questionGenerateWsUrl(): string {
  return `${wsBaseUrl()}/api/question/generate`
}

function questionMimicWsUrl(): string {
  return `${wsBaseUrl()}/api/question/mimic`
}

function questionFollowupWsUrl(): string {
  return `${wsBaseUrl()}/api/question/followup`
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
): () => void {
  const ws = new WebSocket(questionGenerateWsUrl())

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
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return close
}

/** PDF / 已解析目录仿题（/api/question/mimic） */
export function connectQuestionMimic(
  payload: QuestionMimicPayload,
  handlers: QuestionGenHandlers = {},
): () => void {
  const ws = new WebSocket(questionMimicWsUrl())

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
    } catch {
      handlers.onMessage?.({ type: 'parse_error', raw: ev.data })
    }
  }

  ws.onerror = () => handlers.onError?.()
  ws.onclose = () => handlers.onClose?.()

  return close
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
