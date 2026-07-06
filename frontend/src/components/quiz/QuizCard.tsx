import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageSquare, Send, Square, X, ClipboardList } from 'lucide-react'
import type { QuizData, QuizQuestion } from '../../types'
import FormattedMarkdown from '../shared/FormattedMarkdown'
import { connectQuestionFollowup } from '../../services/questionWs'

interface Props {
  quiz: QuizData
  courseId?: string
}

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
  const bottomRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef<(() => void) | null>(null)

  // 卸载时关闭未完成的追问连接，避免 WebSocket 泄漏
  useEffect(() => () => closeRef.current?.(), [])

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
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)

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
            setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 30)
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
      <div className="max-h-52 overflow-y-auto px-3 py-2 space-y-2">
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
        <div ref={bottomRef} />
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
  onSelect,
  onTypedAnswer,
}: {
  q: QuizQuestion
  qIdx: number
  submitted: boolean
  selected: string | undefined
  typedAnswer: string
  onSelect: (idx: number, opt: string) => void
  onTypedAnswer: (idx: number, val: string) => void
}) {
  const [showFollowup, setShowFollowup] = useState(false)
  const isChoice = q.options.length > 0
  const isCorrect = isChoice ? selected === q.answer : null

  // 简答题：提交后且有内容才显示参考答案
  const openAnswerSubmitted = !isChoice && submitted && typedAnswer.trim().length > 0

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
        /* 简答题 作答区域 */
        <div className="space-y-2">
          <textarea
            className="w-full border border-line rounded-[var(--radius)] px-3 py-2 text-sm text-ink placeholder:text-muted bg-surface resize-none focus:outline-none focus:border-ink disabled:bg-surface-2 disabled:text-muted transition"
            rows={3}
            placeholder="在此输入你的答案…"
            value={typedAnswer}
            disabled={submitted}
            onChange={(e) => onTypedAnswer(qIdx, e.target.value)}
          />
          {openAnswerSubmitted && (
            <div className="space-y-2">
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
            isCorrect={isChoice ? isCorrect : null}
            onClose={() => setShowFollowup(false)}
          />
        )}
      </div>
    </div>
  )
}

// ---------- 测验卡片 ----------
export default function QuizCard({ quiz }: Props) {
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({})
  const [typedAnswers, setTypedAnswers] = useState<Record<number, string>>({})
  const [submitted, setSubmitted] = useState(false)

  const choiceQuestions = quiz.questions.filter((q) => q.options.length > 0)
  const correctCount = choiceQuestions.filter(
    (q) => selectedAnswers[quiz.questions.indexOf(q)] === q.answer,
  ).length

  const allChoiceAnswered = choiceQuestions.every(
    (q) => selectedAnswers[quiz.questions.indexOf(q)] !== undefined,
  )

  return (
    <div className="mt-3 border border-line rounded-[var(--radius)] bg-surface-2 overflow-hidden">
      <div className="px-4 py-2.5 bg-canvas border-b border-line">
        <h3 className="text-sm font-semibold text-ink flex items-center gap-1.5">
          <ClipboardList size={15} strokeWidth={1.5} />
          测验题目
          {submitted && choiceQuestions.length > 0 && (
            <span className="text-xs font-normal text-ink-soft ml-2">
              答对了 {correctCount}/{choiceQuestions.length}
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
            onClick={() => setSubmitted(true)}
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
