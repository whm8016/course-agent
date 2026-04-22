/** WebSocket 出题服务：对接后端 /api/question/generate */

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

export interface QuestionGenMessage {
  type: string
  [key: string]: unknown
}

function wsBaseUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

function questionGenerateWsUrl(): string {
  return `${wsBaseUrl()}/api/question/generate`
}

export interface QuestionGenHandlers {
  onMessage?: (msg: QuestionGenMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: () => void
}

/**
 * 建立 WebSocket 连接后立刻发送 payload，返回 close 函数。
 * 注意：必须在 onopen 之后 send，否则连接未就绪会报错。
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
