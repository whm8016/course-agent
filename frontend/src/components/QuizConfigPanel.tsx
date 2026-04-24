import { useState } from 'react'
import { FiChevronDown, FiChevronUp } from 'react-icons/fi'

export interface QuizConfig {
  topic: string
  count: number
  difficulty: string
  questionType: string
  preference: string
}

export const DEFAULT_QUIZ_CONFIG: QuizConfig = {
  topic: '',
  count: 3,
  difficulty: '',
  questionType: '',
  preference: '',
}

interface Props {
  value: QuizConfig
  onChange: (next: QuizConfig) => void
}

export default function QuizConfigPanel({ value, onChange }: Props) {
  const [collapsed, setCollapsed] = useState(true)

  const update = <K extends keyof QuizConfig>(key: K, v: QuizConfig[K]) =>
    onChange({ ...value, [key]: v })

  return (
    <div className="border border-indigo-200 rounded-xl bg-indigo-50/40 overflow-hidden mb-2">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-indigo-700 hover:bg-indigo-50/60 transition"
      >
        <span>📝 出题配置</span>
        {collapsed ? <FiChevronDown size={14} /> : <FiChevronUp size={14} />}
      </button>

      {!collapsed && (
        <div className="px-3 pb-3 space-y-2.5">
          <div>
            <label className="text-[11px] font-medium text-slate-500">知识点 *</label>
            <input
              className="mt-0.5 w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
              placeholder="例如：牛顿第二定律、电路基本定律…"
              value={value.topic}
              onChange={(e) => update('topic', e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <div className="flex-1">
              <label className="text-[11px] font-medium text-slate-500">数量</label>
              <input
                type="number"
                min={1}
                max={20}
                className="mt-0.5 w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
                value={value.count}
                onChange={(e) => update('count', Math.max(1, Math.min(20, Number(e.target.value))))}
              />
            </div>
            <div className="flex-1">
              <label className="text-[11px] font-medium text-slate-500">难度</label>
              <select
                className="mt-0.5 w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                value={value.difficulty}
                onChange={(e) => update('difficulty', e.target.value)}
              >
                <option value="">自动</option>
                <option value="easy">简单</option>
                <option value="medium">中等</option>
                <option value="hard">困难</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="text-[11px] font-medium text-slate-500">题型</label>
              <select
                className="mt-0.5 w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                value={value.questionType}
                onChange={(e) => update('questionType', e.target.value)}
              >
                <option value="">自动</option>
                <option value="choice">选择题</option>
                <option value="true_false">判断题</option>
                <option value="short_answer">简答题</option>
              </select>
            </div>
          </div>
          <div>
            <label className="text-[11px] font-medium text-slate-500">偏好（可选）</label>
            <input
              className="mt-0.5 w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
              placeholder="例如：贴近生活实例、侧重计算…"
              value={value.preference}
              onChange={(e) => update('preference', e.target.value)}
            />
          </div>
        </div>
      )}
    </div>
  )
}
