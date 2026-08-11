import { useState } from 'react'
import { Plus, Send, Trash2 } from 'lucide-react'

export interface OutlineItem {
  title: string
  overview?: string
}

interface Props {
  topic?: string
  sub_topics: OutlineItem[]
  onSubmit: (edited: OutlineItem[]) => void
}

/**
 * 深度研究大纲确认卡片：decompose 后后端暂停（awaiting_user）期间展示。
 * 学生可改 title/overview、增删子主题，点「开始研究」后调 onSubmit（title 为空的项被过滤，
 * 与 deeptutor ResearchOutlineEditor 一致）。提交结果经 submit_user_reply 的 outline 字段回传。
 */
export default function OutlineCard({ topic, sub_topics, onSubmit }: Props) {
  const [items, setItems] = useState<OutlineItem[]>(
    (sub_topics ?? []).map((s) => ({ title: s.title ?? '', overview: s.overview ?? '' })),
  )

  const update = (i: number, key: keyof OutlineItem, value: string) =>
    setItems((prev) => prev.map((it, idx) => (idx === i ? { ...it, [key]: value } : it)))
  const remove = (i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i))
  const add = () => setItems((prev) => [...prev, { title: '', overview: '' }])

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[95%] md:max-w-[80%] rounded-[var(--radius-lg)] px-4 py-3 bg-surface border border-line rounded-bl-[3px] space-y-3">
        <div className="flex items-center gap-2 text-xs font-medium text-info-fg">
          <span>{topic ? `研究大纲：${topic}` : '请确认研究大纲'}</span>
        </div>
        <p className="text-[11px] text-muted leading-relaxed">过目或编辑下面的子主题，确认后开始研究。</p>

        {items.map((it, i) => (
          <div key={i} className="space-y-1.5 rounded-[var(--radius)] border border-line p-2.5 bg-canvas">
            <div className="flex items-center gap-2">
              <input
                value={it.title}
                onChange={(e) => update(i, 'title', e.target.value)}
                placeholder="子主题标题"
                className="flex-1 px-2.5 py-1.5 text-sm font-medium rounded-[var(--radius-sm)] border border-line bg-canvas text-ink outline-none focus:border-ink-soft"
              />
              <button
                onClick={() => remove(i)}
                className="p-1.5 text-muted hover:text-danger-fg transition"
                title="删除该子主题"
              >
                <Trash2 size={15} strokeWidth={1.5} />
              </button>
            </div>
            <textarea
              value={it.overview ?? ''}
              onChange={(e) => update(i, 'overview', e.target.value)}
              placeholder="研究方向说明（可选）"
              rows={2}
              className="w-full px-2.5 py-1.5 text-xs rounded-[var(--radius-sm)] border border-line bg-canvas text-ink-soft outline-none focus:border-ink-soft resize-none leading-relaxed"
            />
          </div>
        ))}

        <button
          onClick={add}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[var(--radius)] border border-line bg-canvas text-ink-soft hover:border-ink transition"
        >
          <Plus size={13} strokeWidth={1.5} />
          增加子主题
        </button>

        <div className="flex justify-end">
          <button
            onClick={() => onSubmit(items.filter((it) => it.title.trim() !== ''))}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[var(--radius)] bg-ink text-canvas hover:opacity-90 transition"
          >
            <Send size={13} strokeWidth={1.5} />
            开始研究
          </button>
        </div>
      </div>
    </div>
  )
}
