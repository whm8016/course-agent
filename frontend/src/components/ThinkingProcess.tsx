import { useState } from 'react'
import { FiChevronDown, FiChevronRight, FiCpu, FiSearch, FiTool } from 'react-icons/fi'
import type { Message } from '../types'

interface Props {
  steps: Message[]
}

export default function ThinkingProcess({ steps }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (steps.length === 0) return null

  return (
    <div className="mb-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-indigo-600 transition"
      >
        {expanded ? <FiChevronDown size={14} /> : <FiChevronRight size={14} />}
        <FiCpu size={13} />
        <span>Agent 思考过程 ({steps.length} 步)</span>
      </button>

      {expanded && (
        <div className="mt-2 ml-4 border-l-2 border-indigo-100 pl-3 space-y-2">
          {steps.map((step, i) => (
            <div key={i} className="text-xs text-slate-500 flex items-start gap-1.5">
              {step.type === 'thinking' && <FiCpu size={12} className="mt-0.5 text-indigo-400 shrink-0" />}
              {step.type === 'tool_call' && <FiTool size={12} className="mt-0.5 text-amber-500 shrink-0" />}
              {step.type === 'tool_result' && <FiSearch size={12} className="mt-0.5 text-green-500 shrink-0" />}
              <div>
                {step.type === 'thinking' && <span>{step.content}</span>}
                {step.type === 'tool_call' && (
                  <span>
                    调用工具 <code className="bg-slate-100 px-1 rounded">{step.metadata?.tool}</code>
                    {step.metadata?.toolInput && (
                      <span className="text-slate-400 ml-1">
                        ({Object.entries(step.metadata.toolInput).map(([k, v]) => `${k}: ${v}`).join(', ')})
                      </span>
                    )}
                  </span>
                )}
                {step.type === 'tool_result' && step.metadata?.chunks && (
                  <span>检索到 {step.metadata.chunks.length} 条相关知识</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
