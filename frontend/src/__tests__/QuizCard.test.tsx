/**
 * QuizCard 提交答案三路收录 + 判分闭环单测（A4）。
 * 覆盖：客观题本地判收录（grade_source=local）/ 简答·coding 先判分再收录
 * （grade/grade_source/feedback 随 payload）/ 判分中态 / 判分失败降级收录
 * （不带 grade_source，后端守卫拦观测）/ coding 骨架预填 / ungraded 展示 /
 * sessionId 缺省静默跳过 / 计分分母=已判分题。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, fireEvent, screen, waitFor } from '@testing-library/react'
import QuizCard from '../components/quiz/QuizCard'
import type { QuizData } from '../types'
import { upsertNotebookEntry, gradeAnswer } from '../services/api'
import type { GradeResponse } from '../services/api'

vi.mock('../services/api', () => ({
  upsertNotebookEntry: vi.fn().mockResolvedValue(undefined),
  gradeAnswer: vi.fn(),
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
      question: '简述光合作用',
      options: [],
      answer: '光能转化为化学能……',
      explanation: '',
      question_id: 'q_2',
      question_type: 'short_answer',
    },
    {
      question: '实现 add(a, b)',
      options: [],
      answer: '```python\ndef add(a, b):\n    return a + b\n```',
      explanation: '',
      question_id: 'q_3',
      question_type: 'coding',
      starter_code: 'def add(a, b):',
      test_cases: ['assert add(1, 2) == 3', 'assert add(0, 0) == 0'],
      language: 'python',
    },
  ],
}

function _grade(grade: GradeResponse['grade'], over: Partial<GradeResponse> = {}): GradeResponse {
  return {
    question_id: 'q',
    grade,
    score: null,
    source: grade === 'ungraded' ? '' : 'llm_judge',
    feedback: '',
    is_correct: grade === 'good' ? true : grade === 'ungraded' ? null : false,
    test_results: [],
    ...over,
  }
}

/** 选完客观题 + 填简答，提交（异步判分链随后由 waitFor 断言） */
function selectTypeAndSubmit(sessionId: string | null, typed = '光合作用是……') {
  const view = render(<QuizCard quiz={quiz} sessionId={sessionId} />)
  fireEvent.click(screen.getByText('B. 2'))
  if (typed !== null) {
    fireEvent.change(screen.getByPlaceholderText('在此输入你的答案…'), {
      target: { value: typed },
    })
  }
  fireEvent.click(screen.getByText('提交答案'))
  return view
}

describe('QuizCard 提交答案三路收录', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(upsertNotebookEntry).mockResolvedValue(undefined)
    vi.mocked(gradeAnswer).mockResolvedValue(_grade('ungraded'))
  })

  it('客观题本地判收录：grade/grade_source=local，无 gradeAnswer 调用', () => {
    selectTypeAndSubmit('sess_1', null) // 不填简答

    expect(upsertNotebookEntry).toHaveBeenCalledWith(
      expect.objectContaining({
        session_id: 'sess_1',
        question_id: 'q_1',
        question_type: 'choice',
        user_answer: 'B',
        is_correct: true,
        grade: 'good',
        grade_source: 'local',
      }),
    )
    expect(gradeAnswer).not.toHaveBeenCalledWith(
      expect.objectContaining({ question_id: 'q_1' }),
    )
  })

  it('简答题先判分再收录：grade=fair/source=llm_judge 随 payload，徽章部分正确', async () => {
    vi.mocked(gradeAnswer).mockResolvedValue(
      _grade('fair', { feedback: '缺原料部分', score: null }),
    )
    const view = selectTypeAndSubmit('sess_2')

    await waitFor(() => {
      expect(upsertNotebookEntry).toHaveBeenCalledWith(
        expect.objectContaining({
          question_id: 'q_2',
          question_type: 'short_answer',
          grade: 'fair',
          grade_source: 'llm_judge',
          feedback: '缺原料部分',
        }),
      )
    })
    await waitFor(() => {
      expect(view.container.textContent).toContain('◐ 部分正确')
    })
  })

  it('coding 题骨架预填 + 判分带 test_cases + 逐用例结果渲染', async () => {
    vi.mocked(gradeAnswer).mockResolvedValue(
      _grade('poor', {
        question_id: 'q_3',
        source: 'sandbox',
        score: 0,
        feedback: '通过 0/2 条用例',
        test_results: [
          { index: 0, passed: false, error: 'AssertionError' },
          { index: 1, passed: false, error: 'AssertionError' },
        ],
      }),
    )
    const view = render(<QuizCard quiz={quiz} sessionId="sess_3" />)

    // 骨架预填：coding textarea 初值 = starter_code
    const codeBox = screen.getByPlaceholderText(
      '在函数骨架基础上补全实现…（提交后将自动运行测试用例判分）',
    ) as HTMLTextAreaElement
    expect(codeBox.value).toBe('def add(a, b):')

    fireEvent.change(codeBox, { target: { value: 'def add(a, b):\n    return a - b' } })
    fireEvent.click(screen.getByText('B. 2'))
    fireEvent.click(screen.getByText('提交答案'))

    await waitFor(() => {
      expect(gradeAnswer).toHaveBeenCalledWith(
        expect.objectContaining({
          question_id: 'q_3',
          test_cases: ['assert add(1, 2) == 3', 'assert add(0, 0) == 0'],
          starter_code: 'def add(a, b):',
        }),
      )
    })
    await waitFor(() => {
      expect(view.container.textContent).toContain('✗ 未通过')
      expect(view.container.textContent).toContain('用例 1')
    })
  })

  it('判分中态：gradeAnswer 未返回时显示"判分中…"', () => {
    vi.mocked(gradeAnswer).mockReturnValue(new Promise(() => {}) as never)
    const view = selectTypeAndSubmit('sess_4')

    expect(view.container.textContent).toContain('判分中…')
  })

  it('判分失败降级：徽章提示 + 仍收录但不带 grade_source（防污染）', async () => {
    vi.mocked(gradeAnswer).mockRejectedValue(new Error('超时'))
    const view = selectTypeAndSubmit('sess_5')

    await waitFor(() => {
      expect(upsertNotebookEntry).toHaveBeenCalledWith(
        expect.objectContaining({
          question_id: 'q_2',
          user_answer: '光合作用是……',
        }),
      )
    })
    const q2Call = vi
      .mocked(upsertNotebookEntry)
      .mock.calls.find((c) => c[0].question_id === 'q_2')
    expect(q2Call?.[0]).not.toHaveProperty('grade_source')
    expect(q2Call?.[0]).not.toHaveProperty('grade')
    await waitFor(() => {
      expect(view.container.textContent).toContain('未能自动判分')
    })
  })

  it('ungraded：徽章"该题未自动判分"，收录不带 grade_source', async () => {
    vi.mocked(gradeAnswer).mockResolvedValue(_grade('ungraded', { feedback: '未能自动判分' }))
    const view = selectTypeAndSubmit('sess_6')

    await waitFor(() => {
      expect(view.container.textContent).toContain('该题未自动判分')
    })
    const q2Call = vi
      .mocked(upsertNotebookEntry)
      .mock.calls.find((c) => c[0].question_id === 'q_2')
    expect(q2Call?.[0].grade_source).toBe('')
  })

  it('sessionId 缺省静默跳过一切请求，判分展示不受影响', () => {
    const view = selectTypeAndSubmit(null)

    expect(gradeAnswer).not.toHaveBeenCalled()
    expect(upsertNotebookEntry).not.toHaveBeenCalled()
    expect(view.container.textContent).toContain('✓ 回答正确')
  })

  it('收录失败不阻塞判分展示', async () => {
    vi.mocked(upsertNotebookEntry).mockRejectedValue(new Error('网络断'))
    vi.mocked(gradeAnswer).mockResolvedValue(_grade('good'))
    const view = selectTypeAndSubmit('sess_7')

    await waitFor(() => {
      expect(view.container.textContent).toContain('✓ 回答正确')
    })
  })

  it('计分单趟不错位：选择题在前时非选择题答对也计数（复审 P1-2）', async () => {
    vi.mocked(gradeAnswer).mockResolvedValue(_grade('good'))
    const view = selectTypeAndSubmit('sess_8')

    await waitFor(() => {
      // 三题全对：choice 本地判对 + 简答 good + coding（骨架预填也算作答）good → 3/3。
      // 错位旧代码对非选择题查 gradeStates[过滤后下标]（choice 的 idle 态）→ 只显示 2/3
      expect(view.container.textContent).toMatch(/答对了 3\/3/)
    })
  })

  it('判分成功但收录失败：徽章保持已判结果，不串扰成 failed（复审 P2-3）', async () => {
    vi.mocked(upsertNotebookEntry).mockRejectedValue(new Error('网络断'))
    vi.mocked(gradeAnswer).mockResolvedValue(_grade('good'))
    const view = selectTypeAndSubmit('sess_9')

    await waitFor(() => {
      expect(view.container.textContent).toContain('✓ 回答正确')
    })
    expect(view.container.textContent).not.toContain('未能自动判分')
  })
})
