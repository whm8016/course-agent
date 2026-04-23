import { useCallback, useEffect, useRef, useState } from 'react'
import { connectQuestionGenerate, type QuestionGenMessage } from '../services/questionWs'
import type { QuizData, QuizQuestion } from '../types'
import QuizCard from './QuizCard'
import FormattedMarkdown from './FormattedMarkdown'

// ---------- 后端 QAPair 的结构 ----------
interface QAPairRaw {
  question?: string
  correct_answer?: string
  explanation?: string
  question_type?: string
  options?: Record<string, string> | null
  concentration?: string
  difficulty?: string
}

interface ResultEntry {
  qa_pair?: QAPairRaw
  success?: boolean
}

// ---------- QAPair → QuizQuestion ----------
function toQuizQuestion(qa: QAPairRaw): QuizQuestion {
  const opts = qa.options
  const type = (qa.question_type ?? '').toLowerCase()

  if (type === 'true_false') {
    return {
      question: qa.question ?? '',
      options: ['A. 正确', 'B. 错误'],
      answer: (qa.correct_answer ?? '').toUpperCase().startsWith('A') ||
              ['true', '正确', '对', 'yes'].includes((qa.correct_answer ?? '').toLowerCase())
        ? 'A'
        : 'B',
      explanation: qa.explanation ?? '',
    }
  }

  if (opts && typeof opts === 'object' && Object.keys(opts).length > 0) {
    const options = Object.entries(opts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([letter, text]) => `${letter}. ${text}`)
    const answer = (qa.correct_answer ?? '').charAt(0).toUpperCase() || 'A'
    return {
      question: qa.question ?? '',
      options,
      answer,
      explanation: qa.explanation ?? '',
    }
  }

  // 简答题 / 无选项：用「点击查看」代替选项卡
  return {
    question: qa.question ?? '',
    options: [],
    answer: qa.correct_answer ?? '',
    explanation: qa.explanation ?? '',
  }
}

// ---------- 简答题卡片 ----------
function OpenQuestionCard({ q, idx }: { q: QuizQuestion; idx: number }) {
  const [revealed, setRevealed] = useState(false)
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden bg-white">
      <div className="px-4 py-3">
        <div className="flex gap-1.5 items-start">
          <span className="text-sm font-semibold text-slate-800 shrink-0 pt-0.5">{idx + 1}.</span>
          <div className="min-w-0 flex-1">
            <FormattedMarkdown
              content={q.question}
              className="markdown-body text-sm text-slate-800 [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_ol]:my-1.5 [&_ul]:my-1.5"
            />
          </div>
        </div>
      </div>
      {!revealed ? (
        <div className="px-4 pb-3">
          <button
            type="button"
            onClick={() => setRevealed(true)}
            className="text-xs px-3 py-1.5 rounded-lg border border-indigo-300 text-indigo-600 hover:bg-indigo-50 transition"
          >
            查看答案
          </button>
        </div>
      ) : (
        <div className="px-4 pb-3 space-y-2">
          <div>
            <p className="text-[11px] font-semibold text-emerald-800 mb-1">参考答案</p>
            <div className="text-emerald-900 bg-emerald-50 px-3 py-2 rounded-lg">
              <FormattedMarkdown
                content={typeof q.answer === 'string' ? q.answer : String(q.answer)}
                className="markdown-body text-xs leading-relaxed"
              />
            </div>
          </div>
          {q.explanation && (
            <div className="bg-slate-50 rounded-lg px-3 py-2 border border-slate-100">
              <p className="text-[11px] font-semibold text-slate-500 mb-1.5">解析</p>
              <FormattedMarkdown content={q.explanation} className="markdown-body text-xs leading-relaxed text-slate-700" />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------- 进度条 ----------
function ProgressBar({ current, total }: { current: number; total: number }) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-500">
        <span>生成进度</span>
        <span>{current}/{total} ({pct}%)</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ---------- 主面板 ----------
interface Props {
  kbName: string
}

type Phase = 'idle' | 'running' | 'done' | 'error'

export default function QuestionGeneratorPanel({ kbName }: Props) {
  const [topic, setTopic] = useState('')
  const [count, setCount] = useState(3)
  const [difficulty, setDifficulty] = useState('')
  const [questionType, setQuestionType] = useState('')
  const [preference, setPreference] = useState('')

  const [phase, setPhase] = useState<Phase>('idle')
  const [statusText, setStatusText] = useState('')
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [questions, setQuestions] = useState<QuizQuestion[]>([])
  const [errorMsg, setErrorMsg] = useState('')

  const closeRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    return () => {
      closeRef.current?.()
    }
  }, [])

  // 把后端 result message 中的 QAPair 追加到 questions 列表
  const handleMessage = useCallback((msg: QuestionGenMessage) => {
    const t = msg.type

    if (t === 'status') {
      setStatusText(String(msg.content ?? ''))
      return
    }

    if (t === 'progress') {
      const stage = String(msg.stage ?? '')
      const cur = Number(msg.current ?? 0)
      const tot = Number(msg.total ?? 0)

      if (stage === 'ideation') {
        setStatusText(`分析知识点，生成题目模板 ${cur}/${tot}…`)
      } else if (stage === 'generation') {
        setProgress({ current: cur, total: tot })
        setStatusText(`正在生成第 ${cur}/${tot} 道题…`)
      } else if (stage === 'complete') {
        setStatusText(`生成完成，共 ${msg.completed ?? cur} 道`)
        setPhase('done')
        closeRef.current?.()
      }
      return
    }

    if (t === 'result') {
      const entry = msg as unknown as ResultEntry
      const qa = entry.qa_pair ?? (msg as { question?: QAPairRaw }).question
      if (qa) {
        setQuestions((prev) => [...prev, toQuizQuestion(qa)])
      }
      return
    }

    if (t === 'complete') {
      setStatusText('生成完成')
      setPhase('done')
      closeRef.current?.()
      return
    }

    if (t === 'error') {
      setErrorMsg(String(msg.content ?? msg))
      setPhase('error')
      closeRef.current?.()
      return
    }
  }, [])

  const handleStart = () => {
    if (!topic.trim() || !kbName) return
    setPhase('running')
    setStatusText('连接中…')
    setProgress({ current: 0, total: count })
    setQuestions([])
    setErrorMsg('')

    const close = connectQuestionGenerate(
      {
        kb_name: kbName,
        count,
        language: 'zh',
        requirement: {
          knowledge_point: topic.trim(),
          preference: preference.trim() || undefined,
          difficulty: difficulty || undefined,
          question_type: questionType || undefined,
        },
      },
      {
        onOpen: () => setStatusText('已连接，出题中…'),
        onMessage: handleMessage,
        onClose: () => {
          if (phase === 'running') setPhase('done')
        },
        onError: () => {
          setErrorMsg('WebSocket 连接失败，请确认后端已启动')
          setPhase('error')
        },
      },
    )

    closeRef.current = close
  }

  const handleStop = () => {
    closeRef.current?.()
    setPhase('idle')
    setStatusText('')
  }

  const handleReset = () => {
    setPhase('idle')
    setStatusText('')
    setQuestions([])
    setErrorMsg('')
    setProgress({ current: 0, total: 0 })
  }

  const isChoice = (q: QuizQuestion) => q.options.length > 0

  // 把所有选择/判断题合成 QuizData 供 QuizCard 用
  const choiceQuestions = questions.filter(isChoice)
  const openQuestions = questions.filter((q) => !isChoice(q))
  const quizData: QuizData = { questions: choiceQuestions }

  const running = phase === 'running'

  return (
    <div className="h-full overflow-y-auto bg-slate-50">
      <div className="max-w-2xl mx-auto px-4 py-6 space-y-5">

        {/* 标题行 */}
        <div className="flex items-center justify-between">
          <h1 className="text-base font-semibold text-slate-800">按知识点出题</h1>
          {(phase === 'done' || phase === 'error') && (
            <button
              type="button"
              onClick={handleReset}
              className="text-xs text-slate-500 hover:text-slate-700 underline"
            >
              重新出题
            </button>
          )}
        </div>

        {/* 无课程提示 */}
        {!kbName && (
          <p className="text-sm text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
            请先在左侧选择一门课程
          </p>
        )}

        {/* 表单 */}
        {phase === 'idle' && (
          <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
            <div>
              <label className="text-xs font-medium text-slate-600">知识点 *</label>
              <input
                className="mt-1 w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                placeholder="例如：牛顿第二定律、动量守恒定律…"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
              />
            </div>
            <div className="flex gap-3">
              <div className="flex-1">
                <label className="text-xs font-medium text-slate-600">数量</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  className="mt-1 w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={count}
                  onChange={(e) => setCount(Math.max(1, Math.min(20, Number(e.target.value))))}
                />
              </div>
              <div className="flex-1">
                <label className="text-xs font-medium text-slate-600">难度</label>
                <select
                  className="mt-1 w-full border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                >
                  <option value="">自动</option>
                  <option value="easy">简单</option>
                  <option value="medium">中等</option>
                  <option value="hard">困难</option>
                </select>
              </div>
              <div className="flex-1">
                <label className="text-xs font-medium text-slate-600">题型</label>
                <select
                  className="mt-1 w-full border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={questionType}
                  onChange={(e) => setQuestionType(e.target.value)}
                >
                  <option value="">自动</option>
                  <option value="choice">选择题</option>
                  <option value="true_false">判断题</option>
                  <option value="short_answer">简答题</option>
                </select>
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-slate-600">偏好（可选）</label>
              <input
                className="mt-1 w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                placeholder="例如：贴近生活实例、侧重计算…"
                value={preference}
                onChange={(e) => setPreference(e.target.value)}
              />
            </div>
            <button
              type="button"
              onClick={handleStart}
              disabled={!topic.trim() || !kbName}
              className="w-full mt-1 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              开始出题
            </button>
          </div>
        )}

        {/* 生成中状态 */}
        {running && (
          <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
            <ProgressBar current={progress.current} total={progress.total} />
            <p className="text-xs text-slate-500 animate-pulse">{statusText}</p>
            <button
              type="button"
              onClick={handleStop}
              className="text-xs px-3 py-1.5 rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50 transition"
            >
              停止
            </button>
          </div>
        )}

        {/* 错误 */}
        {phase === 'error' && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            {errorMsg || '出题失败，请重试'}
          </div>
        )}

        {/* 完成状态栏 */}
        {phase === 'done' && questions.length > 0 && (
          <p className="text-xs text-slate-400">{statusText}（共 {questions.length} 道）</p>
        )}

        {/* 选择 / 判断题：用 QuizCard 渲染 */}
        {choiceQuestions.length > 0 && (
          <QuizCard quiz={quizData} />
        )}

        {/* 简答题：逐题渲染 */}
        {openQuestions.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">简答题</p>
            {openQuestions.map((q, i) => (
              <OpenQuestionCard key={i} q={q} idx={choiceQuestions.length + i} />
            ))}
          </div>
        )}

        {/* 没有题目但已结束 */}
        {phase === 'done' && questions.length === 0 && (
          <p className="text-sm text-slate-400 text-center py-8">没有生成任何题目，请检查知识库是否已建立索引</p>
        )}
      </div>
    </div>
  )
}
