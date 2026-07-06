import { useState } from 'react'
import { ChevronDown, ClipboardList } from 'lucide-react'
import type { QuizConfig } from './types'

// 仅 re-export 类型（type-only 不破坏 fast refresh），保持旧 import 路径兼容
export type { QuizConfig } from './types'

interface Props {
  value: QuizConfig
  onChange: (next: QuizConfig) => void
}

export default function QuizConfigPanel({ value, onChange }: Props) {
  const [collapsed, setCollapsed] = useState(true)

  const update = <K extends keyof QuizConfig>(key: K, v: QuizConfig[K]) =>
    onChange({ ...value, [key]: v })

  const inputCls =
    'mt-0.5 w-full border border-line rounded-[var(--radius)] px-2.5 py-1.5 text-xs bg-surface text-ink placeholder:text-muted focus:outline-none focus:border-ink transition'
  const labelCls = 'text-[11px] font-medium text-muted'

  return (
    <div className="border border-line rounded-[var(--radius)] bg-surface-2 overflow-hidden mb-2">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-ink hover:bg-canvas transition"
      >
        <span className="inline-flex items-center gap-1.5">
          <ClipboardList size={13} strokeWidth={1.5} />
          出题配置
        </span>
        <ChevronDown
          size={14}
          strokeWidth={1.5}
          className={`transition-transform ${collapsed ? '' : 'rotate-180'}`}
        />
      </button>

      {!collapsed && (
        <div className="px-3 pb-3 pt-3 space-y-2.5 border-t border-line">
          <div>
            <label className={labelCls}>知识点 *</label>
            <input
              className={inputCls}
              placeholder="例如：牛顿第二定律、电路基本定律…"
              value={value.topic}
              onChange={(e) => update('topic', e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <div className="flex-1">
              <label className={labelCls}>数量</label>
              <input
                type="number"
                min={1}
                max={20}
                className={inputCls}
                value={value.count}
                onChange={(e) => update('count', Math.max(1, Math.min(20, Number(e.target.value))))}
              />
            </div>
            <div className="flex-1">
              <label className={labelCls}>难度</label>
              <select
                className={inputCls}
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
              <label className={labelCls}>题型</label>
              <select
                className={inputCls}
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
            <label className={labelCls}>偏好（可选）</label>
            <input
              className={inputCls}
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
