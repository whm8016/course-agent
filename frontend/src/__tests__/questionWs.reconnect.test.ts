/**
 * quiz WS 断线恢复单测（questionWs.connectQuestionGenerate 重连包装）。
 * 协议：session 记 turn_id；意外断开带 subscribe_turn{turn_id, after_seq} 重连，
 * after_seq = 已收到的 bus 事件数（session 直发不计入）→ 回放增量不重不漏。
 * FakeWebSocket 桩 + 假时钟驱动退避。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { connectQuestionGenerate } from '../services/questionWs'
import { FakeWebSocket, installFakeWs } from './helpers/wsStubs'

vi.mock('../services/auth', () => ({ getToken: () => 't' }))

beforeEach(() => {
  installFakeWs()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const q = (id: string) =>
  JSON.stringify({ type: 'quiz_question', question: { question_id: id } })

describe('quiz WS 断线恢复', () => {
  it('闪断 → 重连发 subscribe_turn(after_seq=已收数) → 增量回放不重不漏', () => {
    const onMessage = vi.fn()
    connectQuestionGenerate({ course_id: 'c1', question: 'topic' }, { onMessage })
    const ws1 = FakeWebSocket.instances[0]
    ws1.onopen?.()
    expect(ws1.send).toHaveBeenCalledWith(
      JSON.stringify({ course_id: 'c1', question: 'topic' }),
    )
    ws1.onmessage?.({ data: JSON.stringify({ type: 'session', turn_id: 'T1' }) })
    ws1.onmessage?.({ data: q('q1') })
    ws1.onmessage?.({ data: q('q2') })

    ws1.onclose?.() // 意外断开（未主动 close、未 done）
    vi.advanceTimersByTime(1000)

    const ws2 = FakeWebSocket.instances[1]
    ws2.onopen?.()
    expect(ws2.send).toHaveBeenCalledWith(
      JSON.stringify({ type: 'subscribe_turn', turn_id: 'T1', after_seq: 2 }),
    )
    ws2.onmessage?.({ data: q('q3') })
    ws2.onmessage?.({ data: JSON.stringify({ type: 'done' }) })
    ws2.onclose?.()

    // session + q1 + q2 + q3 + done = 5，无重复投递
    expect(onMessage).toHaveBeenCalledTimes(5)
  })

  it('重连后再次收到事件会重置退避计数（连续多次闪断各给 3 次机会）', () => {
    connectQuestionGenerate({ course_id: 'c1', question: 't' }, {})
    const ws1 = FakeWebSocket.instances[0]
    ws1.onopen?.()
    ws1.onmessage?.({ data: JSON.stringify({ type: 'session', turn_id: 'T1' }) })
    ws1.onclose?.()
    vi.advanceTimersByTime(1000)
    const ws2 = FakeWebSocket.instances[1]
    ws2.onopen?.()
    ws2.onmessage?.({ data: q('q1') }) // 有事件到达 → 计数清零
    ws2.onclose?.()
    vi.advanceTimersByTime(1000) // 又从 1s 档开始，而不是 2s
    const ws3 = FakeWebSocket.instances[2]
    ws3.onopen?.()
    expect(ws3.send).toHaveBeenCalledWith(
      JSON.stringify({ type: 'subscribe_turn', turn_id: 'T1', after_seq: 1 }),
    )
  })

  it('重连耗尽（3 次）→ ws_reconnect_failed + onClose，不再无限重试', () => {
    const onMessage = vi.fn()
    const onClose = vi.fn()
    connectQuestionGenerate({ course_id: 'c1', question: 't' }, { onMessage, onClose })
    const ws1 = FakeWebSocket.instances[0]
    ws1.onopen?.()
    ws1.onmessage?.({ data: JSON.stringify({ type: 'session', turn_id: 'T1' }) })
    ws1.onclose?.()

    vi.advanceTimersByTime(1000)
    FakeWebSocket.instances[1].onclose?.()
    vi.advanceTimersByTime(2000)
    FakeWebSocket.instances[2].onclose?.()
    vi.advanceTimersByTime(4000)
    FakeWebSocket.instances[3].onclose?.()
    vi.advanceTimersByTime(30000)

    expect(FakeWebSocket.instances.length).toBe(4) // 原连 + 3 次重连，到此为止
    expect(onMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'ws_reconnect_failed', attempts: 3 }),
    )
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('done/error 后的 close 与主动 close 都不触发重连', () => {
    const onClose = vi.fn()
    const handle = connectQuestionGenerate({ course_id: 'c1', question: 't' }, { onClose })
    const ws1 = FakeWebSocket.instances[0]
    ws1.onopen?.()
    ws1.onmessage?.({ data: JSON.stringify({ type: 'session', turn_id: 'T1' }) })
    ws1.onmessage?.({ data: JSON.stringify({ type: 'done' }) })
    ws1.onclose?.()
    vi.advanceTimersByTime(30000)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(FakeWebSocket.instances.length).toBe(1)

    handle.close()
    expect(ws1.close).toHaveBeenCalled()
  })

  it('尚未拿到 turn_id 就断开（初始失败）：不重连，走 onClose', () => {
    const onClose = vi.fn()
    connectQuestionGenerate({ course_id: 'c1', question: 't' }, { onClose })
    const ws1 = FakeWebSocket.instances[0]
    ws1.onopen?.()
    ws1.onclose?.()
    vi.advanceTimersByTime(30000)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(FakeWebSocket.instances.length).toBe(1)
  })
})
