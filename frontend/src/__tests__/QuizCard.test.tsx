/**
 * QuizCard 提交答案三路收录 + 判分闭环单测（A4）。
 * 覆盖：客观题本地判收录（grade_source=local）/ 简答·coding 先判分再收录
 * （grade/grade_source/feedback 随 payload）/ 判分中态 / 判分失败降级收录
 * （不带 grade_source，后端守卫拦观测）/ coding 骨架预填 / ungraded 展示 /
 * sessionId 缺省静默跳过 / 计分分母=已判分题。
 */
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest'
import { render, fireEvent, screen, waitFor, act } from '@testing-library/react'
import QuizCard from '../components/quiz/QuizCard'
import type { QuizData } from '../types'
import { upsertNotebookEntry, gradeAnswer } from '../services/api'
import type { GradeResponse } from '../services/api'

vi.mock('../services/api', () => ({
  upsertNotebookEntry: vi.fn().mockResolvedValue(undefined),
  gradeAnswer: vi.fn(),
}))

// 追问面板 WS：捕获 handlers 供测试注入流式 token
const followupState = vi.hoisted(() => ({
  handlers: {} as { onMessage?: (m: Record<string, unknown>) => void },
}))
vi.mock('../services/questionWs', () => ({
  connectQuestionFollowup: vi.fn(
    (_payload: unknown, handlers: typeof followupState.handlers) => {
      followupState.handlers = handlers
      return vi.fn()
    },
  ),
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
function selectTypeAndSubmit(sessionId: string | null, typed: string | null = '光合作用是……') {
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

// 两个 describe 共用：清空所有 mock 调用记录（含追问块模块级的 scrollIntoView spy）
beforeEach(() => {
  vi.clearAllMocks()
})

describe('QuizCard 提交答案三路收录', () => {
  beforeEach(() => {
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

describe('QuizCard 追问面板流式滚动', () => {
  // jsdom 不实现 scrollIntoView，stub 成 spy：旧实现每个 token 调一次，可数
  const scrollIntoView = vi.fn()

  beforeAll(() => {
    Object.defineProperty(Element.prototype, 'scrollIntoView', {
      value: scrollIntoView,
      configurable: true,
    })
  })

  // 用例内中途换 fake timers，断言抛错也不能把假计时器泄漏给后续用例
  afterEach(() => vi.useRealTimers())

  /** 模拟用户上拉脱离贴底：jsdom 无布局，手动把容器报告成"离底 300px"再派发 scroll */
  function detachFromBottom(panel: HTMLDivElement) {
    Object.defineProperty(panel, 'scrollHeight', { get: () => 500, configurable: true })
    Object.defineProperty(panel, 'clientHeight', { get: () => 200, configurable: true })
    fireEvent.scroll(panel)
  }

  /** 打开第 1 题追问面板，并劫持面板滚动容器的 scrollTop 赋值（jsdom 无布局，以此观测跟随） */
  function openFollowupPanel() {
    render(<QuizCard quiz={quiz} sessionId={null} />)
    fireEvent.click(screen.getAllByText('追问')[0])
    // 结构锚点：输入框所在 footer 的上一个兄弟即面板滚动容器（不锚 max-h-52 这类纯样式类）
    const panel = screen.getByPlaceholderText('输入追问内容…').parentElement
      ?.previousElementSibling as HTMLDivElement
    expect(panel).not.toBeNull()
    const setScrollTop = vi.fn()
    Object.defineProperty(panel, 'scrollTop', {
      get: () => 0,
      set: setScrollTop,
      configurable: true,
    })
    return { panel, setScrollTop }
  }

  function sendFollowup() {
    fireEvent.change(screen.getByPlaceholderText('输入追问内容…'), {
      target: { value: '为什么' },
    })
    fireEvent.click(screen.getByTitle('发送'))
  }

  it('流式跟随只动面板自身 scrollTop，绝不 scrollIntoView（那会拖动整页祖先链）', async () => {
    const { setScrollTop } = openFollowupPanel()
    // fake timers：若回归出旧式"30/50ms 定时器里 scrollIntoView"的实现，
    // 冲掉 pending 定时器会立刻暴露，不用真等 120ms
    vi.useFakeTimers()
    sendFollowup()

    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '因' })
      followupState.handlers.onMessage?.({ type: 'token', content: '为' })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(150)
    })
    // 此后只有同步断言，无定时器依赖——恢复真实计时器交给 afterEach 兜底

    // 不许碰 scrollIntoView：连带滚动祖先链（整个聊天页被反复压到底）+ smooth 高频排队
    expect(scrollIntoView).not.toHaveBeenCalled()
    // 跟随发生在面板容器自身
    expect(setScrollTop).toHaveBeenCalled()
  })

  it('用户在面板内上拉脱离贴底后，后续 token 不再拽回', async () => {
    const { panel, setScrollTop } = openFollowupPanel()
    sendFollowup()

    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '答' })
    })
    expect(setScrollTop).toHaveBeenCalled()
    setScrollTop.mockClear()

    // 脱离贴底：后续 token 不跟随
    detachFromBottom(panel)

    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '案' })
      followupState.handlers.onMessage?.({ type: 'token', content: '续' })
    })

    expect(setScrollTop).not.toHaveBeenCalled()
    expect(scrollIntoView).not.toHaveBeenCalled()
  })

  it('用户拉回底部后恢复跟随', () => {
    const { panel, setScrollTop } = openFollowupPanel()
    sendFollowup()
    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '答' })
    })
    setScrollTop.mockClear()

    // 脱离贴底：后续 token 不跟随
    detachFromBottom(panel)
    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '案' })
    })
    expect(setScrollTop).not.toHaveBeenCalled()

    // 拉回贴底（离底 20px）：恢复跟随
    Object.defineProperty(panel, 'scrollTop', {
      get: () => 480,
      set: setScrollTop,
      configurable: true,
    })
    fireEvent.scroll(panel)
    setScrollTop.mockClear()
    act(() => {
      followupState.handlers.onMessage?.({ type: 'token', content: '续' })
    })
    expect(setScrollTop).toHaveBeenCalled()
    expect(scrollIntoView).not.toHaveBeenCalled()
  })
})
