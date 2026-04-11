import { useState } from 'react'
import { FiBookOpen, FiChevronDown, FiChevronRight } from 'react-icons/fi'
import type { RagChunk } from '../types'

interface Props {
  chunks: RagChunk[]
}

export default function SourceCard({ chunks }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (chunks.length === 0) return null

  return (
    <div className="mt-3 border border-slate-200 rounded-xl bg-slate-50/80 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-600 hover:bg-slate-100 transition"
      >
        <FiBookOpen size={13} className="text-indigo-500" />
        <span className="font-medium">知识来源 ({chunks.length})</span>
        {expanded ? <FiChevronDown size={14} className="ml-auto" /> : <FiChevronRight size={14} className="ml-auto" />}
      </button>

      {expanded && (
        <div className="border-t border-slate-200 divide-y divide-slate-100">
          {chunks.map((chunk, i) => (
            <div key={i} className="px-3 py-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-slate-700">{chunk.source}</span>
                <span className="text-xs text-indigo-500 font-mono">
                  {(chunk.score * 100).toFixed(0)}% 匹配
                </span>
              </div>
              <p className="text-xs text-slate-500 line-clamp-3 leading-relaxed">
                {chunk.content}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
