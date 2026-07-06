import { MessageSquare, Plus, Trash2 } from 'lucide-react'
import type { Session } from '../../types'

interface Props {
  sessions: Session[]
  activeSessionId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
}

export default function SessionList({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  onDelete,
}: Props) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-3 mb-2">
        <h2 className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted">对话</h2>
        <button
          onClick={onCreate}
          className="p-1 -mr-1 rounded-[var(--radius-sm)] text-muted hover:text-ink hover:bg-surface-2 transition"
          title="新对话"
          aria-label="新对话"
        >
          <Plus size={14} strokeWidth={1.5} />
        </button>
      </div>

      {sessions.length === 0 && <p className="text-xs text-muted px-3">暂无对话记录</p>}

      {sessions.map((s) => {
        const active = activeSessionId === s.id
        return (
          <div
            key={s.id}
            className={`group flex items-center gap-2 px-3 py-2 rounded-[var(--radius)] cursor-pointer transition-colors text-sm ${
              active
                ? 'bg-surface-2 text-ink font-medium'
                : 'text-ink-soft hover:bg-surface-2 hover:text-ink'
            }`}
            onClick={() => onSelect(s.id)}
          >
            <MessageSquare size={14} strokeWidth={1.5} className="shrink-0" />
            <span className="flex-1 truncate">{s.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDelete(s.id)
              }}
              className="opacity-0 group-hover:opacity-100 p-0.5 rounded-[var(--radius-sm)] text-muted hover:text-danger-fg transition"
              aria-label="删除对话"
            >
              <Trash2 size={12} strokeWidth={1.5} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
