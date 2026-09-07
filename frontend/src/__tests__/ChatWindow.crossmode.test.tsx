/**
 * ChatWindow 跨模式记忆 payload 单测（Bug A 前端半边）。
 * 场景：同会话先聊了两轮（fetchMessages 拉回历史），切「出题」/「深度研究」后发起——
 * connectQuestionGenerate / connectDeepResearch 收到的 payload 必须带刚才的对话
 * （history，OpenAI message 格式）与 session_id，后端才能以"背景摘录"形态带入
 * quiz/research 管线（M-6 清空对话轮次后仍有指代对象）。
 * services/api 与 questionWs 全 mock；能力切换走真实 UI（cap 菜单）。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import ChatWindow from '../components/chat/ChatWindow'
import { fetchMessages } from '../services/api'
import { connectQuestionGenerate, connectDeepResearch } from '../services/questionWs'

/** 每条 research 假连接的句柄：测试经 handlers.onMessage 注入事件、断言 close */
const wsState = vi.hoisted(() => ({
  research: [] as {
    handlers: { onMessage?: (m: Record<string, unknown>) => void }
    close: ReturnType<typeof vi.fn>
  }[],
}))

vi.mock('../services/api', () => ({
  chatStream: vi.fn(),
  uploadImage: vi.fn(),
  fetchMessages: vi.fn(),
  saveMessage: vi.fn().mockResolvedValue(undefined),
  createSession: vi.fn(),
  updateSessionMode: vi.fn().mockResolvedValue(undefined),
  fetchLlmProfilesSelectable: vi.fn().mockResolvedValue({ profiles: [], active: '' }),
  fetchMyLlmProvider: vi.fn().mockResolvedValue(null),
  deleteMyLlmProvider: vi.fn(),
  requestAnswerNow: vi.fn(),
}))

vi.mock('../services/questionWs', () => ({
  connectQuestionGenerate: vi.fn(() => ({ close: vi.fn(), send: vi.fn() })),
  connectDeepResearch: vi.fn(
    (payload: unknown, handlers: wsState['research'][number]['handlers']) => {
      const conn = { handlers, close: vi.fn() }
      wsState.research.push(conn)
      return { close: conn.close, send: vi.fn() }
    },
  ),
}))

/** 切模式前同会话的两轮讨论（后端 /sessions/:id/messages 拉回的行） */
const PRIOR_TURNS = [
  { role: 'user', content: '戴维南等效怎么用？', msg_type: 'text' },
  { role: 'assistant', content: '把复杂电路换成电源+内阻再看。', msg_type: 'text' },
]
const EXPECTED_HISTORY = PRIOR_TURNS.map((m) => ({ role: m.role, content: m.content }))

function renderChat() {
  return render(
    <ChatWindow
      courseId="c1"
      courseName="电路原理"
      sessionId="s1"
      onSessionCreated={vi.fn()}
    />,
  )
}

/** 切能力菜单（移动端+桌面端两份菜单同时渲染，取第一份）并等输入框 placeholder 切换 */
async function switchCapAndWait(label: RegExp, placeholder: RegExp) {
  fireEvent.click(screen.getAllByTitle('切换能力')[0])
  fireEvent.click(screen.getAllByRole('button', { name: label })[0])
  await waitFor(() => expect(screen.getByPlaceholderText(placeholder)).toBeTruthy())
}

/** 等 mount 时 fetchMessages 的历史渲染进消息列表（保证 messages state 已填充） */
async function waitHistoryLoaded() {
  await waitFor(() => expect(screen.getAllByText(/戴维南等效/).length).toBeGreaterThan(0))
}

describe('ChatWindow 跨模式记忆：quiz/research payload 带 history + session_id', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    wsState.research.length = 0
    vi.mocked(fetchMessages).mockResolvedValue(PRIOR_TURNS as never)
  })

  it('出题：connectQuestionGenerate 收到刚才的对话与会话 id', async () => {
    renderChat()
    await waitHistoryLoaded()
    await switchCapAndWait(/^出题/, /输入知识点直接出题/)

    const ta = screen.getByPlaceholderText(/输入知识点直接出题/) as HTMLTextAreaElement
    fireEvent.change(ta, { target: { value: '针对刚才的讨论出几道题' } })
    fireEvent.keyDown(ta, { key: 'Enter' })

    await waitFor(() => expect(connectQuestionGenerate).toHaveBeenCalled())
    const payload = vi.mocked(connectQuestionGenerate).mock.calls[0][0]
    expect(payload.session_id).toBe('s1')
    expect(payload.history).toEqual(EXPECTED_HISTORY)
  })

  it('深度研究：connectDeepResearch 收到刚才的对话与会话 id', async () => {
    renderChat()
    await waitHistoryLoaded()
    await switchCapAndWait(/深度研究/, /输入研究主题/)

    const ta = screen.getByPlaceholderText(/输入研究主题/) as HTMLTextAreaElement
    fireEvent.change(ta, { target: { value: '针对刚才的讨论做份研究' } })
    fireEvent.keyDown(ta, { key: 'Enter' })

    await waitFor(() => expect(connectDeepResearch).toHaveBeenCalled())
    const payload = vi.mocked(connectDeepResearch).mock.calls[0][0]
    expect(payload.session_id).toBe('s1')
    expect(payload.history).toEqual(EXPECTED_HISTORY)
  })

  /** 走到"大纲确认卡挂起"状态：发研究 → 注入 stage_start(init)+outline_card */
  async function showOutlineCard() {
    const ta = screen.getByPlaceholderText(/输入研究主题/) as HTMLTextAreaElement
    fireEvent.change(ta, { target: { value: '戴维南专题研究' } })
    fireEvent.keyDown(ta, { key: 'Enter' })
    await waitFor(() => expect(wsState.research.length).toBe(1))
    const conn = wsState.research[0]
    act(() => {
      conn.handlers.onMessage?.({ type: 'stage_start', stage: 'init', research_id: 'r1' })
      conn.handlers.onMessage?.({
        type: 'outline_card',
        topic: '戴维南',
        sub_topics: [{ title: '子题A', overview: '' }],
      })
    })
    await screen.findByText(/研究大纲/)
    return conn
  }

  it('大纲确认卡：点取消 → 连接关闭、卡片撤下、不触发断线重连（P0 回归钉）', async () => {
    renderChat()
    await waitHistoryLoaded()
    await switchCapAndWait(/深度研究/, /输入研究主题/)

    const conn = await showOutlineCard()

    fireEvent.click(screen.getByRole('button', { name: '取消' }))

    await waitFor(() => expect(conn.close).toHaveBeenCalled())
    // 取消走主动 close（close 内已解绑 onclose）→ 不允许再开第二条连接重发卡片
    expect(connectDeepResearch).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/研究大纲/)).toBeNull()
  })

  it('大纲确认卡挂起时切换会话：卡片一并撤下（P1 回归钉）', async () => {
    const view = renderChat()
    await waitHistoryLoaded()
    await switchCapAndWait(/深度研究/, /输入研究主题/)
    await showOutlineCard()

    view.rerender(
      <ChatWindow
        courseId="c1"
        courseName="电路原理"
        sessionId="s2"
        onSessionCreated={vi.fn()}
      />,
    )

    await waitFor(() => expect(screen.queryByText(/研究大纲/)).toBeNull())
  })
})
