import { useState } from 'react'
import { BookOpen, ChevronDown } from 'lucide-react'
import type { RagChunk } from '../../types'

interface Props {
  chunks: RagChunk[]
}

export default function SourceCard({ chunks }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (chunks.length === 0) return null

  return (
    <div className="mt-3 border border-line rounded-[var(--radius)] bg-surface-2 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-ink-soft hover:bg-canvas transition"
      >
        <BookOpen size={13} strokeWidth={1.5} className="text-ink-soft" />
        <span className="font-medium">知识来源 ({chunks.length})</span>
        <ChevronDown
          size={14}
          strokeWidth={1.5}
          className={`ml-auto transition-transform ${expanded ? '' : '-rotate-90'}`}
        />
      </button>

      {expanded && (
        <div className="border-t border-line divide-y divide-line">
          {chunks.map((chunk, i) => (
            <div key={i} className="px-3 py-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-ink">{chunk.source}</span>
                <span className="text-xs text-info-fg font-mono">
                  {(chunk.score * 100).toFixed(0)}% 匹配
                </span>
              </div>
              <p className="text-xs text-muted line-clamp-3 leading-relaxed">{chunk.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
