import { useCallback, useRef, useState } from 'react'
import { FiMessageSquare, FiSend, FiX } from 'react-icons/fi'
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
    const m = /^([A-D])[\.\):]\s*(.*)$/i.exec(line.trim())
    if (m) out[m[1].toUpperCase()] = m[2]
  }
  return out
}

function FollowupPanel({ question, questionIndex, userAnswer, isCorrect, onClose }: FollowupPanelProps) {
  const [messages, setMessages] = useState<FollowupMsg[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

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
      }
    }

    connectQuestionFollowup(
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

  return (
    <div className="mt-3 border border-indigo-100 rounded-xl bg-indigo-50/30 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-indigo-100 bg-indigo-50/50">
        <span className="text-xs font-medium text-indigo-700">追问</span>
        <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-600">
          <FiX size={13} />
        </button>
      </div>
      <div className="max-h-52 overflow-y-auto px-3 py-2 space-y-2">
        {messages.length === 0 && (
          <p className="text-[11px] text-slate-400 text-center py-2">有问题？在这里追问吧</p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`text-xs rounded-lg px-2.5 py-1.5 ${
              m.role === 'user'
                ? 'bg-indigo-100 text-indigo-800 ml-6'
                : 'bg-white border border-slate-200 text-slate-700 mr-6'
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
            <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="flex items-center gap-2 px-3 py-2 border-t border-indigo-100">
        <input
          className="flex-1 text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-300 bg-white"
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
          onClick={() => void handleSend()}
          disabled={!input.trim() || loading}
          className="p-1.5 rounded-lg bg-indigo-600 text-white disabled:opacity-40 transition"
        >
          <FiSend size={12} />
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
    <div className="border-b border-slate-100 last:border-0 pb-5 last:pb-0">
      {/* 题干 */}
      <div className="mb-2.5 flex gap-1.5 items-start">
        <span className="text-sm font-semibold text-slate-800 shrink-0 pt-0.5">{qIdx + 1}.</span>
        <div className="min-w-0 flex-1">
          <FormattedMarkdown
            content={q.question}
            className="markdown-body text-sm text-slate-800 [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0"
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

              let cls = 'w-full text-left px-3 py-2 rounded-lg text-sm border transition '
              if (!submitted) {
                cls += isSelected
                  ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                  : 'border-slate-200 hover:border-slate-300 text-slate-700'
              } else if (isAnswer) {
                cls += 'border-green-400 bg-green-50 text-green-700'
              } else if (isSelected && !isCorrect) {
                cls += 'border-red-400 bg-red-50 text-red-700'
              } else {
                cls += 'border-slate-200 text-slate-400'
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
                className={`px-3 py-2 rounded-lg text-xs font-medium ${
                  isCorrect ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'
                }`}
              >
                {isCorrect ? '✓ 回答正确' : `✗ 正确答案是 ${q.answer}`}
              </div>
              {q.explanation && (
                <div className="px-3 py-2 rounded-lg border border-slate-100 bg-slate-50/80">
                  <p className="text-[11px] font-semibold text-slate-500 mb-1.5">解析</p>
                  <FormattedMarkdown
                    content={q.explanation}
                    className="markdown-body text-xs leading-relaxed text-slate-700"
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
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-50 disabled:text-slate-400 transition"
            rows={3}
            placeholder="在此输入你的答案…"
            value={typedAnswer}
            disabled={submitted}
            onChange={(e) => onTypedAnswer(qIdx, e.target.value)}
          />
          {openAnswerSubmitted && (
            <div className="space-y-2">
              <div className="text-emerald-900 bg-emerald-50 px-3 py-2 rounded-lg">
                <p className="text-[11px] font-semibold text-emerald-700 mb-1">参考答案</p>
                <FormattedMarkdown
                  content={typeof q.answer === 'string' ? q.answer : String(q.answer)}
                  className="markdown-body text-xs leading-relaxed"
                />
              </div>
              {q.explanation && (
                <div className="bg-slate-50 rounded-lg px-3 py-2 border border-slate-100">
                  <p className="text-[11px] font-semibold text-slate-500 mb-1">解析</p>
                  <FormattedMarkdown
                    content={q.explanation}
                    className="markdown-body text-xs leading-relaxed text-slate-700"
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
            className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-500 transition"
          >
            <FiMessageSquare size={12} />
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
export default function QuizCard({ quiz, courseId: _courseId }: Props) {
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
    <div className="mt-3 border border-indigo-200 rounded-xl bg-indigo-50/50 overflow-hidden">
      <div className="px-4 py-2.5 bg-indigo-100/60 border-b border-indigo-200">
        <h3 className="text-sm font-semibold text-indigo-800 flex items-center gap-1.5">
          📝 测验题目
          {submitted && choiceQuestions.length > 0 && (
            <span className="text-xs font-normal text-indigo-600 ml-2">
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
            className="w-full py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            提交答案
          </button>
        </div>
      )}
    </div>
  )
}
