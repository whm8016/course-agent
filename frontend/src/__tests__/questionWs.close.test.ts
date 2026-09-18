/**
 * questionWs close() 语义单测（P0 修复回归钉）。
 * 主动 close 必须先解绑 ws.onclose/onerror 再调 ws.close()——浏览器对主动关闭同样
 * 派发 close 事件，不解绑会被上层当"意外断开"（research 的 onClose 带
 * resume_research_id 自动重连，把刚取消的大纲确认卡原样弹回）。
 * 服务端真断开（未主动 close）时回调照常触达，重连逻辑不受影响。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { connectDeepResearch, connectQuestionGenerate } from '../services/questionWs'

vi.mock('../services/auth', () => ({ getToken: () => 't' }))

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static CONNECTING = 0
  static OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  readyState = 1 // OPEN
  close = vi.fn()
  send = vi.fn()
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
}

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('WsHandle.close 主动关闭语义', () => {
  it('connectDeepResearch：close 后 onclose/onerror 已解绑，主动关闭事件不触达 handlers', () => {
    const onClose = vi.fn()
    const onError = vi.fn()
    const handle = connectDeepResearch(
      { course_id: 'c1', question: 'q' },
      { onClose, onError },
    )
    const ws = FakeWebSocket.instances[0]

    handle.close()

    expect(ws.close).toHaveBeenCalled()
    expect(ws.onclose).toBeNull()
    expect(ws.onerror).toBeNull()
    // 模拟浏览器对主动关闭派发 close/error 事件：属性已 null，handlers 不会被调
    ws.onclose?.()
    ws.onerror?.()
    expect(onClose).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
  })

  it('connectQuestionGenerate：同款解绑语义', () => {
    const onClose = vi.fn()
    const handle = connectQuestionGenerate(
      { course_id: 'c1', question: 'q' },
      { onClose },
    )
    const ws = FakeWebSocket.instances[0]

    handle.close()

    expect(ws.close).toHaveBeenCalled()
    expect(ws.onclose).toBeNull()
  })

  it('服务端断开（未主动 close）：onClose 照常触达（重连逻辑不受影响）', () => {
    const onClose = vi.fn()
    connectDeepResearch({ course_id: 'c1', question: 'q' }, { onClose })
    const ws = FakeWebSocket.instances[0]

    ws.onclose?.() // 浏览器在非主动关闭时派发，此时 handler 仍在

    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
