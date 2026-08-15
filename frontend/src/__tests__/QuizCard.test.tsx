/**
 * QuizCard 提交答案自动收录笔记本单测。
 * 修复背景：NotebookPage 文案承诺"出题/答题后自动收录"，但提交按钮原本只
 * setSubmitted(true) 零请求——判分在前端本地做、后端永远收不到作答，
 * notebook_entries 恒 0 行。现提交时逐题 POST upsert（best-effort）。
 * 覆盖：客观题收录且 payload 正确 / 简答题零调用 / sessionId 缺省静默跳过 /
 * 收录失败不阻塞判分展示。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, fireEvent, screen } from '@testing-library/react'
import QuizCard from '../components/quiz/QuizCard'
import type { QuizData } from '../types'
import { upsertNotebookEntry } from '../services/api'

vi.mock('../services/api', () => ({
  upsertNotebookEntry: vi.fn().mockResolvedValue(undefined),
}))

const quiz: QuizData = {
  questions: [
    {
      question: '1+1=?',
      options: ['A. 1', 'B. 2'],
      answer: 'B',
      explanation: '基础加法',
      question_id: 'q_1',
      question_type: 'choice',
      difficulty: 'easy',
    },
    {
      question: '谈谈你的理解',
      options: [],
      answer: '参考答案',
      explanation: '',
      question_id: 'q_2',
      question_type: 'short_answer',
    },
  ],
}

/** 选完客观题并提交（选择 B 使 is_correct=true），返回 view 供后续断言 */
function selectAndSubmit(sessionId: string | null) {
  const view = render(<QuizCard quiz={quiz} sessionId={sessionId} />)
  fireEvent.click(screen.getByText('B. 2'))
  fireEvent.click(screen.getByText('提交答案'))
  return view
}

describe('QuizCard 提交答案自动收录', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(upsertNotebookEntry).mockResolvedValue(undefined)
  })

  it('提交后客观题逐题 upsert（payload 带幂等键与判分），简答题零调用', () => {
    selectAndSubmit('sess_1')

    expect(upsertNotebookEntry).toHaveBeenCalledTimes(1) // 简答题（options 空）不收录
    expect(upsertNotebookEntry).toHaveBeenCalledWith(
      expect.objectContaining({
        session_id: 'sess_1',
        question_id: 'q_1',
        question: '1+1=?',
        question_type: 'choice',
        options: { A: '1', B: '2' },
        correct_answer: 'B',
        explanation: '基础加法',
        difficulty: 'easy',
        user_answer: 'B',
        is_correct: true,
      }),
    )
  })

  it('答错也如实收录（is_correct=false）', () => {
    const view = render(<QuizCard quiz={quiz} sessionId="sess_2" />)
    fireEvent.click(screen.getByText('A. 1'))
    fireEvent.click(screen.getByText('提交答案'))

    expect(upsertNotebookEntry).toHaveBeenCalledWith(
      expect.objectContaining({ user_answer: 'A', is_correct: false }),
    )
    expect(view.container.textContent).toContain('✗ 正确答案是 B')
  })

  it('sessionId 缺省（会话创建失败的边缘态）静默跳过收录，判分展示不受影响', () => {
    const view = selectAndSubmit(null)

    expect(upsertNotebookEntry).not.toHaveBeenCalled()
    expect(view.container.textContent).toContain('✓ 回答正确')
  })

  it('收录失败（网络断/后端重启）不阻塞判分展示、不抛未捕获异常', () => {
    vi.mocked(upsertNotebookEntry).mockRejectedValue(new Error('网络断'))
    const view = selectAndSubmit('sess_3')

    expect(upsertNotebookEntry).toHaveBeenCalledTimes(1)
    expect(view.container.textContent).toContain('✓ 回答正确')
  })
})
