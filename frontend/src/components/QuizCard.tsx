import { useState } from 'react'
import type { QuizData } from '../types'
import FormattedMarkdown from './FormattedMarkdown'

interface Props {
  quiz: QuizData
}

export default function QuizCard({ quiz }: Props) {
  const [answers, setAnswers] = useState<Record<number, string>>({})
  const [submitted, setSubmitted] = useState(false)

  const handleSelect = (qIdx: number, option: string) => {
    if (submitted) return
    setAnswers((prev) => ({ ...prev, [qIdx]: option }))
  }

  const handleSubmit = () => {
    setSubmitted(true)
  }

  const correctCount = quiz.questions.filter(
    (q, i) => answers[i] === q.answer
  ).length

  return (
    <div className="mt-3 border border-indigo-200 rounded-xl bg-indigo-50/50 overflow-hidden">
      <div className="px-4 py-2.5 bg-indigo-100/60 border-b border-indigo-200">
        <h3 className="text-sm font-semibold text-indigo-800 flex items-center gap-1.5">
          📝 课堂测验
          {submitted && (
            <span className="text-xs font-normal text-indigo-600 ml-2">
              得分：{correctCount}/{quiz.questions.length}
            </span>
          )}
        </h3>
      </div>

      <div className="p-4 space-y-5">
        {quiz.questions.map((q, qIdx) => {
          const selected = answers[qIdx]
          const isCorrect = selected === q.answer

          return (
            <div key={qIdx}>
              <div className="mb-2 flex gap-1.5 items-start">
                <span className="text-sm font-semibold text-slate-800 shrink-0 pt-0.5">{qIdx + 1}.</span>
                <div className="min-w-0 flex-1">
                  <FormattedMarkdown
                    content={q.question}
                    className="markdown-body text-sm text-slate-800 [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_ol]:my-1.5 [&_ul]:my-1.5"
                  />
                </div>
              </div>
              <div className="space-y-1.5">
                {q.options.map((opt) => {
                  const optLetter = opt.charAt(0)
                  const isSelected = selected === optLetter
                  const isAnswer = q.answer === optLetter

                  let className = 'w-full text-left px-3 py-2 rounded-lg text-sm border transition '
                  if (!submitted) {
                    className += isSelected
                      ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                      : 'border-slate-200 hover:border-slate-300 text-slate-700'
                  } else if (isAnswer) {
                    className += 'border-green-400 bg-green-50 text-green-700'
                  } else if (isSelected && !isCorrect) {
                    className += 'border-red-400 bg-red-50 text-red-700'
                  } else {
                    className += 'border-slate-200 text-slate-400'
                  }

                  return (
                    <button
                      key={opt}
                      onClick={() => handleSelect(qIdx, optLetter)}
                      className={className}
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
                    className={`px-3 py-2 rounded-lg text-xs font-medium ${isCorrect ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'}`}
                  >
                    {isCorrect ? '✓ 回答正确！' : `✗ 正确答案是 ${q.answer}`}
                  </div>
                  {q.explanation && (
                    <div className="px-1 py-2 rounded-lg border border-slate-100 bg-slate-50/80">
                      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide mb-1.5">解析</p>
                      <FormattedMarkdown
                        content={q.explanation}
                        className="markdown-body text-xs leading-relaxed text-slate-700"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {!submitted && (
        <div className="px-4 pb-4">
          <button
            onClick={handleSubmit}
            disabled={Object.keys(answers).length < quiz.questions.length}
            className="w-full py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            提交答案
          </button>
        </div>
      )}
    </div>
  )
}
