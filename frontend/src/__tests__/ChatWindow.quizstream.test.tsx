/**
 * ChatWindow 出题流式事件消费单测：
 * - quiz_question_error 显式提示（不静默吞题）
 * - 全部题目失败 → 不产出空测验卡（提交按钮可点的假成功）
 * - 正常 done → 测验卡落消息列表（判断题 A/B 归一联动）
 * services/api 与 questionWs 全 mock，事件经捕获的 handlers 注入。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import ChatWindow from '../components/chat/ChatWindow'
import { fetchMessages } from '../services/api'

type Handlers = {
  onMessage?: (m: Record<string, unknown>) => void
  onClose?: () => void
  onError?: () => void
}

/** 每条 quiz 假连接的句柄 */
const quizState = vi.hoisted(() => ({
  conns: [] as { handlers: Handlers; close: ReturnType<typeof vi.fn> }[],
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
  connectQuestionGenerate: vi.fn(
    (_payload: unknown, handlers: (typeof quizState)['conns'][number]['handlers']) => {
      const conn = { handlers, close: vi.fn() }
      quizState.conns.push(conn)
      return { close: conn.close, send: vi.fn() }
    },
  ),
  connectDeepResearch: vi.fn(() => ({ close: vi.fn(), send: vi.fn() })),
}))

function renderChat() {
  return render(
    <ChatWindow courseId="c1" courseName="电路原理" sessionId="s1" onSessionCreated={vi.fn()} />,
  )
}

/** 切到出题模式、输入知识点、回车发起，返回本条连接 */
async function startQuiz(topic: string) {
  fireEvent.click(screen.getAllByTitle('切换能力')[0])
  fireEvent.click(screen.getAllByRole('button', { name: /^出题/ })[0])
  const ta = (await screen.findByPlaceholderText(/输入知识点直接出题/)) as HTMLTextAreaElement
  fireEvent.change(ta, { target: { value: topic } })
  fireEvent.keyDown(ta, { key: 'Enter' })
  await waitFor(() => expect(quizState.conns.length).toBe(1))
  return quizState.conns[0]
}

describe('ChatWindow 出题流式事件', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    quizState.conns.length = 0
    vi.mocked(fetchMessages).mockResolvedValue([] as never)
  })

  it('quiz_question_error：界面出现「已跳过」提示，不再静默吞题', async () => {
    renderChat()
    const conn = await startQuiz('叠加定理')

    act(() => {
      conn.handlers.onMessage?.({ type: 'quiz_question_error', index: 2, reason: 'critic 未通过' })
    })
    expect(await screen.findByText(/第 3 题生成失败已跳过/)).toBeTruthy()
    expect(screen.getByText(/critic 未通过/)).toBeTruthy()
  })

  it('全部失败：done 不产出空测验卡，报错可见（错误不随气泡卸载）', async () => {
    renderChat()
    const conn = await startQuiz('叠加定理')

    act(() => {
      conn.handlers.onMessage?.({ type: 'quiz_question_error', index: 0, reason: '生成失败' })
      conn.handlers.onMessage?.({ type: 'done' })
    })
    // 错误横幅保持可见（此前 error 长在气泡里、quizStreaming 一关就一闪而过）
    expect(await screen.findByText(/题目全部生成失败/)).toBeTruthy()
    // 不出现可提交的空测验卡
    expect(screen.queryByRole('button', { name: '提交答案' })).toBeNull()
    expect(conn.close).toHaveBeenCalled()
  })

  it('正常 done：测验卡落消息列表，判断题归一成 A/B 选项', async () => {
    renderChat()
    const conn = await startQuiz('叠加定理')

    act(() => {
      conn.handlers.onMessage?.({
        type: 'quiz_question',
        question: {
          question_id: 'q1',
          question_type: 'concept',
          question: '叠加定理只适用于线性电路',
          correct_answer: 'true',
          explanation: '非线性不适用',
        },
      })
      conn.handlers.onMessage?.({ type: 'done' })
    })
    // 测验卡出现：题面 + 判断题 A/B 选项 + 提交按钮
    expect(await screen.findByText(/叠加定理只适用于线性电路/)).toBeTruthy()
    expect(screen.getByText('A. 正确')).toBeTruthy()
    expect(screen.getByText('B. 错误')).toBeTruthy()
    expect(screen.getByRole('button', { name: '提交答案' })).toBeTruthy()
  })
})
