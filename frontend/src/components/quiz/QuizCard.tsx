import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageSquare, Send, Square, X, ClipboardList, Loader2 } from 'lucide-react'
import type { QuizData, QuizQuestion } from '../../types'
import FormattedMarkdown from '../shared/FormattedMarkdown'
import { connectQuestionFollowup } from '../../services/questionWs'
import { upsertNotebookEntry, gradeAnswer } from '../../services/api'
import type { GradeResponse } from '../../services/api'

interface Props {
  quiz: QuizData
  courseId?: string
  /** 所属会话 id：提交答案自动收录笔记本的幂等键之一 */
  sessionId?: string | null
}

/** 非选择题判分状态（A4）：提交后逐题 pending → done/failed */
type GradeState =
  | { status: 'idle' }
  | { status: 'pending' }
  | { status: 'done'; result: GradeResponse }
  | { status: 'failed' }

// ---------- Follow-up chat (WebSocket /api/question/followup) ----------
interface FollowupMsg {
  role: 'user' | 'assistant'
  content: string
}

interface FollowupPanelProps {
  question: QuizQuestion
  questionIndex: number
  userAnswer: string
  isCorrect: boolean | null
  onClose: () => void
}

function quizOptionsToRecord(options: string[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (const line of options) {
    // 字符类内的 . 和 ) 是字面量，无需转义
    const m = /^([A-D])[.):]\s*(.*)$/i.exec(line.trim())
    if (m) out[m[1].toUpperCase()] = m[2]
  }
  return out
}

function FollowupPanel({ question, questionIndex, userAnswer, isCorrect, onClose }: FollowupPanelProps) {
  const [messages, setMessages] = useState<FollowupMsg[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const closeRef = useRef<(() => void) | null>(null)

  // 卸载时关闭未完成的追问连接，避免 WebSocket 泄漏
  useEffect(() => () => closeRef.current?.(), [])

  // 贴底判定（与 ChatWindow 同口径 40px）：用户上拉脱离即停止自动跟随
  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setIsAtBottom((prev) => (prev !== near ? near : prev))
  }, [])

  // 流式新内容到达时，仅在贴底期间跟随，直接赋值本容器 scrollTop（瞬时、只动面板）。
  // 不用 bottomRef.scrollIntoView({smooth})——那会连带滚动祖先链（整个聊天页被反复
  // 压到底），且每个 token 排一次 smooth 动画与用户手动滚动打架（ChatWindow 已踩过同坑）
  useEffect(() => {
    if (!isAtBottom) return
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, loading, isAtBottom])

  const questionType = question.options?.length ? 'choice' : 'written'

  const buildQuestionContext = useCallback((): Record<string, unknown> => {
    return {
      question_id: `q_${questionIndex + 1}`,
      question_type: questionType,
      question: question.question,
      options: quizOptionsToRecord(question.options ?? []),
      correct_answer: question.answer,
      explanation: question.explanation,
      user_answer: userAnswer,
      is_correct: isCorrect,
    }
  }, [question, questionIndex, questionType, userAnswer, isCorrect])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: FollowupMsg = { role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setIsAtBottom(true) // 自己刚发的话出现在底部，恢复跟随（只滚面板自身）

    const historyContext =
      messages.length > 0
        ? messages.map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n')
        : ''

    let answer = ''
    let finished = false
    const finish = () => {
      if (!finished) {
        finished = true
        setLoading(false)
        closeRef.current = null
      }
    }

    closeRef.current = connectQuestionFollowup(
      {
        question_context: buildQuestionContext(),
        history_context: historyContext,
        user_message: text,
        language: 'zh',
      },
      {
        onMessage: (event) => {
          const t = event.type
          if (t === 'token') {
            if (closeRef.current === null) return // 已停止 / 已完成，忽略迟到的 token
            answer += String(event.content ?? '')
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { ...last, content: answer }]
              }
              return [...prev, { role: 'assistant', content: answer }]
            })
          } else if (t === 'answer') {
            answer = String(event.content ?? answer)
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { ...last, content: answer }]
              }
              return [...prev, { role: 'assistant', content: answer }]
            })
          } else if (t === 'error') {
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: `Error: ${String(event.content ?? '')}` },
            ])
          }
        },
        onError: () => {
          setMessages((prev) => [...prev, { role: 'assistant', content: 'Connection failed.' }])
          finish()
        },
        onComplete: finish,
        onClose: finish,
      },
    )
  }, [input, loading, messages, buildQuestionContext])

  // 停止追问流式：关闭 WS + 标记已停止（与主窗口 handleStop 行为一致）
  const handleStop = useCallback(() => {
    closeRef.current?.()
    closeRef.current = null
    setLoading(false)
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (last?.role === 'assistant' && last.content.trim()) {
        return [...prev.slice(0, -1), { ...last, content: `${last.content}\n\n_（已停止生成）_` }]
      }
      return [...prev, { role: 'assistant', content: '_（已停止生成）_' }]
    })
  }, [])

  return (
    <div className="mt-3 border border-line rounded-[var(--radius)] bg-surface-2 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-line bg-canvas">
        <span className="text-xs font-medium text-ink-soft">追问</span>
        <button type="button" onClick={onClose} className="text-muted hover:text-ink transition">
          <X size={13} strokeWidth={1.5} />
        </button>
      </div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-52 overflow-y-auto px-3 py-2 space-y-2"
      >
        {messages.length === 0 && (
          <p className="text-[11px] text-muted text-center py-2">有问题？在这里追问吧</p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`text-xs rounded-[var(--radius)] px-2.5 py-1.5 ${
              m.role === 'user'
                ? 'bg-surface-2 text-ink ml-6'
                : 'bg-surface border border-line text-ink-soft mr-6'
            }`}
          >
            <FormattedMarkdown
              content={m.content}
              className="markdown-body [&_p]:my-0.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0"
            />
          </div>
        ))}
        {loading && (
          <div className="flex gap-1 items-center ml-1 pb-1">
            <span className="w-1.5 h-1.5 bg-ink-soft rounded-full animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 bg-ink-soft rounded-full animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 bg-ink-soft rounded-full animate-bounce" />
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 px-3 py-2 border-t border-line">
        <input
          className="flex-1 text-xs border border-line rounded-[var(--radius)] px-2.5 py-1.5 bg-surface text-ink placeholder:text-muted focus:outline-none focus:border-ink"
          placeholder="输入追问内容…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void handleSend()
            }
          }}
          disabled={loading}
        />
        <button
          type="button"
          onClick={loading ? handleStop : () => void handleSend()}
          disabled={!loading && !input.trim()}
          title={loading ? '停止' : '发送'}
          className={`p-1.5 rounded-[var(--radius)] text-white disabled:opacity-40 disabled:cursor-not-allowed transition ${
            loading ? 'bg-danger-fg hover:opacity-90' : 'bg-accent hover:bg-accent-2'
          }`}
        >
          {loading ? <Square size={12} strokeWidth={1.5} /> : <Send size={12} strokeWidth={1.5} />}
        </button>
      </div>
    </div>
  )
}

// ---------- 题目渲染 + 追问入口 ----------
function QuestionItem({
  q,
  qIdx,
  submitted,
  selected,
  typedAnswer,
  gradeState,
  onSelect,
  onTypedAnswer,
}: {
  q: QuizQuestion
  qIdx: number
  submitted: boolean
  selected: string | undefined
  typedAnswer: string
  gradeState?: GradeState
  onSelect: (idx: number, opt: string) => void
  onTypedAnswer: (idx: number, val: string) => void
}) {
  const [showFollowup, setShowFollowup] = useState(false)
  const isChoice = q.options.length > 0
  const isCoding = (q.question_type ?? '').toLowerCase() === 'coding'
  const isCorrect = isChoice ? selected === q.answer : null

  // 简答题：提交后且有内容才显示参考答案
  const openAnswerSubmitted = !isChoice && submitted && typedAnswer.trim().length > 0

  // 判分徽章（A4）：pending 转圈 / good✓ / fair◐（含通过率）/ poor✗ / ungraded 未判
  const gradeBadge = () => {
    if (isChoice || !submitted) return null
    const gs = gradeState ?? { status: 'idle' as const }
    if (gs.status === 'idle') return null
    if (gs.status === 'pending') {
      return (
        <div className="flex items-center gap-1.5 text-xs text-ink-soft bg-surface-2 border border-line px-3 py-2 rounded-[var(--radius)]">
          <Loader2 size={13} className="animate-spin" strokeWidth={1.5} />
          判分中…
        </div>
      )
    }
    if (gs.status === 'failed') {
      return (
        <div className="text-xs text-warn-fg bg-warn-bg px-3 py-2 rounded-[var(--radius)]">
          未能自动判分（网络/超时），参考答案如下
        </div>
      )
    }
    const r = gs.result
    if (r.grade === 'ungraded') {
      return (
        <div className="text-xs text-muted bg-surface-2 border border-line px-3 py-2 rounded-[var(--radius)]">
          该题未自动判分{r.feedback ? `（${r.feedback}）` : ''}
        </div>
      )
    }
    const label =
      r.grade === 'good'
        ? '✓ 回答正确'
        : r.grade === 'fair'
          ? `◐ 部分正确${r.score != null ? `（通过率 ${Math.round(r.score * 100)}%）` : ''}`
          : '✗ 未通过'
    const cls =
      r.grade === 'good'
        ? 'text-ok-fg bg-ok-bg'
        : r.grade === 'fair'
          ? 'text-warn-fg bg-warn-bg'
          : 'text-danger-fg bg-danger-bg'
    return (
      <div className={`px-3 py-2 rounded-[var(--radius)] text-xs font-medium ${cls}`}>
        <p>{label}</p>
        {r.feedback && <p className="font-normal mt-1 opacity-90">{r.feedback}</p>}
        {r.test_results.length > 0 && (
          <ul className="font-normal mt-1.5 space-y-0.5 opacity-90">
            {r.test_results.map((t) => (
              <li key={t.index}>
                {t.passed ? '✓' : '✗'} 用例 {t.index + 1}
                {!t.passed && t.error ? `：${t.error.slice(0, 80)}` : ''}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return (
    <div className="border-b border-line last:border-0 pb-5 last:pb-0">
      {/* 题干 */}
      <div className="mb-2.5 flex gap-1.5 items-start">
        <span className="text-sm font-semibold text-ink shrink-0 pt-0.5">{qIdx + 1}.</span>
        <div className="min-w-0 flex-1">
          <FormattedMarkdown
            content={q.question}
            className="markdown-body text-sm text-ink [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0"
          />
        </div>
      </div>

      {isChoice ? (
        /* 单选 / 多选题 */
        <>
          <div className="space-y-1.5">
            {q.options.map((opt) => {
              const optLetter = opt.charAt(0)
              const isSelected = selected === optLetter
              const isAnswer = q.answer === optLetter

              let cls = 'w-full text-left px-3 py-2 rounded-[var(--radius)] text-sm border transition '
              if (!submitted) {
                cls += isSelected
                  ? 'border-ink bg-surface-2 text-ink'
                  : 'border-line hover:border-ink-soft text-ink-soft'
              } else if (isAnswer) {
                cls += 'border-ok-fg bg-ok-bg text-ok-fg'
              } else if (isSelected && !isCorrect) {
                cls += 'border-danger-fg bg-danger-bg text-danger-fg'
              } else {
                cls += 'border-line text-muted'
              }

              return (
                <button
                  key={opt}
                  onClick={() => onSelect(qIdx, optLetter)}
                  className={cls}
                  disabled={submitted}
                >
                  {opt}
                </button>
              )
            })}
          </div>
          {submitted && (
            <div className="mt-2 space-y-2">
              <div
                className={`px-3 py-2 rounded-[var(--radius)] text-xs font-medium ${
                  isCorrect ? 'bg-ok-bg text-ok-fg' : 'bg-warn-bg text-warn-fg'
                }`}
              >
                {isCorrect ? '✓ 回答正确' : `✗ 正确答案是 ${q.answer}`}
              </div>
              {q.explanation && (
                <div className="px-3 py-2 rounded-[var(--radius)] border border-line bg-surface-2">
                  <p className="text-[11px] font-semibold text-muted mb-1.5">解析</p>
                  <FormattedMarkdown
                    content={q.explanation}
                    className="markdown-body text-xs leading-relaxed text-ink-soft"
                  />
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        /* 简答 / coding 作答区域 */
        <div className="space-y-2">
          <textarea
            className={`w-full border border-line rounded-[var(--radius)] px-3 py-2 text-sm text-ink placeholder:text-muted bg-surface resize-none focus:outline-none focus:border-ink disabled:bg-surface-2 disabled:text-muted transition ${
              isCoding ? 'font-mono text-[13px] leading-relaxed' : ''
            }`}
            rows={isCoding ? 10 : 3}
            spellCheck={!isCoding}
            placeholder={
              isCoding
                ? '在函数骨架基础上补全实现…（提交后将自动运行测试用例判分）'
                : '在此输入你的答案…'
            }
            value={typedAnswer}
            disabled={submitted}
            onChange={(e) => onTypedAnswer(qIdx, e.target.value)}
          />
          {openAnswerSubmitted && (
            <div className="space-y-2">
              {gradeBadge()}
              <div className="text-ok-fg bg-ok-bg px-3 py-2 rounded-[var(--radius)]">
                <p className="text-[11px] font-semibold text-ok-fg mb-1">参考答案</p>
                <FormattedMarkdown
                  content={typeof q.answer === 'string' ? q.answer : String(q.answer)}
                  className="markdown-body text-xs leading-relaxed"
                />
              </div>
              {q.explanation && (
                <div className="bg-surface-2 rounded-[var(--radius)] px-3 py-2 border border-line">
                  <p className="text-[11px] font-semibold text-muted mb-1">解析</p>
                  <FormattedMarkdown
                    content={q.explanation}
                    className="markdown-body text-xs leading-relaxed text-ink-soft"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 追问入口 */}
      <div className="mt-2.5">
        {!showFollowup ? (
          <button
            type="button"
            onClick={() => setShowFollowup(true)}
            className="inline-flex items-center gap-1 text-[11px] text-muted hover:text-ink transition"
          >
            <MessageSquare size={12} strokeWidth={1.5} />
            追问
          </button>
        ) : (
          <FollowupPanel
            question={q}
            questionIndex={qIdx}
            userAnswer={isChoice ? (selected ?? '') : typedAnswer}
            isCorrect={
              isChoice
                ? isCorrect
                : gradeState?.status === 'done'
                  ? gradeState.result.is_correct
                  : null
            }
            onClose={() => setShowFollowup(false)}
          />
        )}
      </div>
    </div>
  )
}

// ---------- 测验卡片 ----------
export default function QuizCard({ quiz, sessionId }: Props) {
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({})
  const [typedAnswers, setTypedAnswers] = useState<Record<number, string>>({})
  const [gradeStates, setGradeStates] = useState<Record<number, GradeState>>({})
  const [submitted, setSubmitted] = useState(false)

  // coding 题预填函数签名骨架（学生 textarea 自带骨架 = 完整可运行代码契约，
  // 提交后沙箱直接跑，不用再前插 starter_code）
  useEffect(() => {
    setTypedAnswers((prev) => {
      const next = { ...prev }
      let changed = false
      quiz.questions.forEach((q, i) => {
        if (
          next[i] === undefined &&
          (q.question_type ?? '').toLowerCase() === 'coding' &&
          q.starter_code
        ) {
          next[i] = q.starter_code
          changed = true
        }
      })
      return changed ? next : prev
    })
  }, [quiz.questions])

  // 提交答案（A4 三路）：choice/判断 → 本地判 + 收录；简答/coding → 先判分再收录
  // （判分结果带 grade/grade_source，后端守卫据此放行 mastery 观测回流）。
  // 判分失败也收录题面作答，但不带 grade_source（守卫拦观测，防污染）。
  // 失败静默，不阻塞判分展示。判断题已被 toQuizQuestion 归一为 options 形式。
  const handleSubmit = () => {
    setSubmitted(true)
    if (!sessionId) return
    quiz.questions.forEach((q, i) => {
      const qid = q.question_id || `q_${i + 1}` // chat 模式 quiz 事件无后端 id 时兜底
      const base = {
        session_id: sessionId,
        question_id: qid,
        question: q.question,
        question_type: q.question_type || 'choice',
        correct_answer: String(q.answer),
        explanation: q.explanation,
        difficulty: q.difficulty || '',
      }

      if (q.options.length > 0) {
        // 本地判分路径：is_correct 前端算（选项唯一确定），grade_source="local"
        void upsertNotebookEntry({
          ...base,
          options: quizOptionsToRecord(q.options),
          user_answer: selectedAnswers[i] ?? '',
          is_correct: selectedAnswers[i] === q.answer,
          grade: selectedAnswers[i] === q.answer ? 'good' : 'poor',
          grade_source: 'local',
        }).catch(() => {})
        return
      }

      // 简答 / coding：判分（coding→沙箱，简答→LLM）→ 收录 → 渲染
      const userAnswer = typedAnswers[i] ?? ''
      setGradeStates((prev) => ({ ...prev, [i]: { status: 'pending' } }))
      void gradeAnswer({
        ...base,
        question_type: q.question_type || 'written',
        user_answer: userAnswer,
        test_cases: q.test_cases ?? [],
        starter_code: q.starter_code ?? '',
        language: q.language ?? '',
      })
        .then((result) => {
          // 判分成功即定格 done——收录失败不该把已判的结果串扰成 failed 态，
          // 也不该丢掉判分产物（复审 P2-3：两段 catch 分开）
          setGradeStates((prev) => ({ ...prev, [i]: { status: 'done', result } }))
          void upsertNotebookEntry({
            ...base,
            question_type: q.question_type || 'written',
            user_answer: userAnswer,
            is_correct: result.is_correct ?? false,
            grade: result.grade === 'ungraded' ? '' : result.grade,
            grade_source: result.source || '',
            feedback: result.feedback,
            score: result.score,
          }).catch(() => {})
        })
        .catch(() => {
          setGradeStates((prev) => ({ ...prev, [i]: { status: 'failed' } }))
          void upsertNotebookEntry({
            ...base,
            question_type: q.question_type || 'written',
            user_answer: userAnswer,
          }).catch(() => {})
        })
    })
  }

  // 计分口径（A4）：分母 = 已判分题。单趟遍历同一原始下标 i——
  // 先 filter 再用过滤后下标查 gradeStates 会错位（复审 P1-2）
  const choiceQuestions = quiz.questions.filter((q) => q.options.length > 0)
  let gradedCount = 0
  let correctCount = 0
  quiz.questions.forEach((q, i) => {
    if (q.options.length > 0) {
      gradedCount += 1
      if (selectedAnswers[i] === q.answer) correctCount += 1
      return
    }
    const gs = gradeStates[i]
    if (gs?.status === 'done' && gs.result.grade !== 'ungraded') {
      gradedCount += 1
      if (gs.result.grade === 'good') correctCount += 1
    }
  })

  const allChoiceAnswered = choiceQuestions.every(
    (q) => selectedAnswers[quiz.questions.indexOf(q)] !== undefined,
  )

  return (
    <div className="mt-3 border border-line rounded-[var(--radius)] bg-surface-2 overflow-hidden">
      <div className="px-4 py-2.5 bg-canvas border-b border-line">
        <h3 className="text-sm font-semibold text-ink flex items-center gap-1.5">
          <ClipboardList size={15} strokeWidth={1.5} />
          测验题目
          {submitted && gradedCount > 0 && (
            <span className="text-xs font-normal text-ink-soft ml-2">
              答对了 {correctCount}/{gradedCount}
            </span>
          )}
        </h3>
      </div>

      <div className="p-4 space-y-0">
        {quiz.questions.map((q, qIdx) => (
          <QuestionItem
            key={qIdx}
            q={q}
            qIdx={qIdx}
            submitted={submitted}
            selected={selectedAnswers[qIdx]}
            typedAnswer={typedAnswers[qIdx] ?? ''}
            gradeState={gradeStates[qIdx] ?? { status: 'idle' }}
            onSelect={(idx, opt) => {
              if (submitted) return
              setSelectedAnswers((prev) => ({ ...prev, [idx]: opt }))
            }}
            onTypedAnswer={(idx, val) => {
              if (submitted) return
              setTypedAnswers((prev) => ({ ...prev, [idx]: val }))
            }}
          />
        ))}
      </div>

      {!submitted && (
        <div className="px-4 pb-4">
          <button
            onClick={handleSubmit}
            disabled={choiceQuestions.length > 0 && !allChoiceAnswered}
            className="w-full py-2 rounded-[var(--radius)] bg-accent hover:bg-accent-2 text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            提交答案
          </button>
        </div>
      )}
    </div>
  )
}
