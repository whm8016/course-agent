import { useState } from 'react'
import { HelpCircle, Send } from 'lucide-react'

export interface AskUserQuestion {
  id: string
  prompt: string
  options?: string[]
}

export interface AskUserReply {
  questionId: string
  text: string
}

interface Props {
  intro?: string
  questions: AskUserQuestion[]
  onSubmit: (answers: AskUserReply[]) => void
}

/**
 * ask_user 工具触发的问题卡片：后端 loop 暂停等回复期间展示。
 * 有 options 的题→选项按钮单选；无 options→文本框。提交后调 onSubmit（未答给空串=跳过）。
 */
export default function AskUserCard({ intro, questions, onSubmit }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const set = (id: string, text: string) => setAnswers((p) => ({ ...p, [id]: text }))

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[95%] md:max-w-[80%] rounded-[var(--radius-lg)] px-4 py-3 bg-surface border border-line rounded-bl-[3px] space-y-3">
        <div className="flex items-center gap-2 text-xs font-medium text-info-fg">
          <HelpCircle size={14} strokeWidth={1.5} />
          <span>{intro || '需要你补充一点信息，我才能继续'}</span>
        </div>

        {questions.map((q) => (
          <div key={q.id} className="space-y-1.5">
            <p className="text-sm text-ink leading-relaxed">{q.prompt}</p>
            {q.options && q.options.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {q.options.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => set(q.id, opt)}
                    className={`px-3 py-1.5 text-xs rounded-[var(--radius)] border transition ${
                      answers[q.id] === opt
                        ? 'bg-ink text-canvas border-ink'
                        : 'bg-canvas text-ink border-line hover:border-ink-soft'
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            ) : (
              <input
                value={answers[q.id] ?? ''}
                onChange={(e) => set(q.id, e.target.value)}
                placeholder="输入你的回答…"
                className="w-full px-3 py-1.5 text-sm rounded-[var(--radius)] border border-line bg-canvas text-ink outline-none focus:border-ink-soft"
              />
            )}
          </div>
        ))}

        <div className="flex justify-end">
          <button
            onClick={() =>
              onSubmit(questions.map((q) => ({ questionId: q.id, text: answers[q.id] ?? '' })))
            }
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[var(--radius)] bg-ink text-canvas hover:opacity-90 transition"
          >
            <Send size={13} strokeWidth={1.5} />
            提交
          </button>
        </div>
      </div>
    </div>
  )
}
