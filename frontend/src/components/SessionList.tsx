import { FiMessageSquare, FiPlus, FiTrash2 } from 'react-icons/fi'
import type { Session } from '../types'

interface Props {
  sessions: Session[]
  activeSessionId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
}

export default function SessionList({ sessions, activeSessionId, onSelect, onCreate, onDelete }: Props) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-3 mb-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          对话
        </h2>
        <button
          onClick={onCreate}
          className="p-1 rounded text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition"
          title="新对话"
        >
          <FiPlus size={14} />
        </button>
      </div>

      {sessions.length === 0 && (
        <p className="text-xs text-slate-400 px-3">暂无对话记录</p>
      )}

      {sessions.map((s) => (
        <div
          key={s.id}
          className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all text-sm ${
            activeSessionId === s.id
              ? 'bg-indigo-50 text-indigo-700'
              : 'text-slate-600 hover:bg-slate-50'
          }`}
          onClick={() => onSelect(s.id)}
        >
          <FiMessageSquare size={14} className="shrink-0" />
          <span className="flex-1 truncate">{s.title}</span>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete(s.id)
            }}
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-400 hover:text-red-500 transition"
          >
            <FiTrash2 size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}
