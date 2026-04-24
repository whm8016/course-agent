import { useCallback, useRef, useState } from 'react'
import { FiMessageSquare, FiSend, FiX } from 'react-icons/fi'
import type { QuizData, QuizQuestion } from '../types'
import FormattedMarkdown from './FormattedMarkdown'
import { chatStream } from '../services/api'

interface Props {
  quiz: QuizData
  courseId?: string
}

// ---------- 追问对话 ----------
interface FollowupMsg {
  role: 'user' | 'assistant'
  content: string
}

interface FollowupPanelProps {
  question: QuizQuestion
  userAnswer: string       // 用户作答内容（选项字母 or 简答文本）
  isCorrect: boolean | null
  courseId?: string
  onClose: () => void
}

function FollowupPanel({ question, userAnswer, isCorrect, courseId, onClose }: FollowupPanelProps) {
  const [messages, setMessages] = useState<FollowupMsg[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // 题目上下文字符串，作为每次调用时 history 的"前缀"
  const contextPrefix = [
    `[题目上下文]`,
    `题目：${question.question}`,
    question.options?.length ? `选项：${question.options.join('，')}` : '',
    `正确答案：${question.answer}`,
    question.explanation ? `解析：${question.explanation}` : '',
    userAnswer
      ? `学生作答：${userAnswer}，${
          isCorrect === true ? '回答正确' : isCorrect === false ? '回答错误' : '未作答'
        }`
      : '',
  ]
    .filter(Boolean)
    .join('\n')

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: FollowupMsg = { role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)

    // 正确的 history 拼法：
    // 第1条：把题目上下文拼进第一条 user 消息（相当于 system prompt）
    // 后续：追问对话的完整历史
    // 最后：当前用户输入
    const history: { role: string; content: string }[] = [
      { role: 'user', content: `${contextPrefix}\n\n${messages.length === 0 ? text : messages[0]?.content ?? text}` },
      ...messages.slice(1).map((m) => ({ role: m.role, content: m.content })),
    ]
    // 若已有对话，最后追加当前用户输入
    if (messages.length > 0) {
      history.push({ role: 'user', content: text })
    }

    const controller = new AbortController()
    abortRef.current = controller
    let answer = ''

    await chatStream(
      courseId ?? '',
      // 发给后端的 message 就是用户的追问文本
      text,
      // history 传已有对话（不含当前这条），后端会把 message 附加在最后
      history.slice(0, -1),
      undefined,
      undefined,
      'chat',
      controller.signal,
      (event) => {
        if (event.type === 'token') {
          answer += event.content ?? ''
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { ...last, content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
          setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 30)
        } else if (event.type === 'answer') {
          answer = event.content ?? ''
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { ...last, content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
        }
      },
      (err) => {
        setMessages((prev) => [...prev, { role: 'assistant', content: `出错了: ${err}` }])
      },
    )

    setLoading(false)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, loading, messages, courseId, contextPrefix])

  return (
    <div className="mt-3 border border-indigo-100 rounded-xl bg-indigo-50/30 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-indigo-100 bg-indigo-50/50">
        <span className="text-xs font-medium text-indigo-700">💬 追问</span>
        <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-600">
          <FiX size={13} />
        </button>
      </div>
      <div className="max-h-52 overflow-y-auto px-3 py-2 space-y-2">
        {messages.length === 0 && (
          <p className="text-[11px] text-slate-400 text-center py-2">
            可以问关于这道题的任何问题
          </p>
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
          placeholder="输入追问…"
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

// ---------- 单题（含回答框 + 追问） ----------
function QuestionItem({
  q,
  qIdx,
  submitted,
  selected,
  typedAnswer,
  onSelect,
  onTypedAnswer,
  courseId,
}: {
  q: QuizQuestion
  qIdx: number
  submitted: boolean
  selected: string | undefined
  typedAnswer: string
  onSelect: (idx: number, opt: string) => void
  onTypedAnswer: (idx: number, val: string) => void
  courseId?: string
}) {
  const [showFollowup, setShowFollowup] = useState(false)
  const isChoice = q.options.length > 0
  const isCorrect = isChoice ? selected === q.answer : null

  // 简答题：提交后判断是否和参考答案接近（宽松匹配）
  const openAnswerSubmitted = !isChoice && submitted && typedAnswer.trim().length > 0

  return (
    <div className="border-b border-slate-100 last:border-0 pb-5 last:pb-0">
      {/* 题目文本 */}
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
        /* 选择 / 判断题 */
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
                {isCorrect ? '✓ 回答正确！' : `✗ 正确答案是 ${q.answer}`}
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
        /* 简答题 —— 有回答框 */
        <div className="space-y-2">
          <textarea
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-50 disabled:text-slate-400 transition"
            rows={3}
            placeholder="在此输入你的回答…"
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

      {/* 追问区 */}
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
            userAnswer={isChoice ? (selected ?? '') : typedAnswer}
            isCorrect={isChoice ? isCorrect : null}
            courseId={courseId}
            onClose={() => setShowFollowup(false)}
          />
        )}
      </div>
    </div>
  )
}

// ---------- 主卡片 ----------
export default function QuizCard({ quiz, courseId }: Props) {
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
          📝 课堂测验
          {submitted && choiceQuestions.length > 0 && (
            <span className="text-xs font-normal text-indigo-600 ml-2">
              选择题得分：{correctCount}/{choiceQuestions.length}
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
            courseId={courseId}
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
