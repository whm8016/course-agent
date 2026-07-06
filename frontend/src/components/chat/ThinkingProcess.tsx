import { useState, useEffect, useRef } from 'react'
import { ChevronDown, BrainCircuit, Database, Eye, MessageSquare, Loader2 } from 'lucide-react'
import type { Message } from '../../types'

interface Props {
  steps: Message[]
  isStreaming?: boolean
}

type StageId = 'thinking' | 'retrieving' | 'observing' | 'responding'

interface StageGroup {
  stage: StageId
  label: string
  startedAt: number | null
  completedAt: number | null
  content: string
  state: 'running' | 'complete'
}

const STAGE_LABELS: Record<StageId, string> = {
  thinking: '分析问题',
  retrieving: '检索知识图谱',
  observing: '整理证据',
  responding: '生成回答',
}

const STAGE_ICONS: Record<StageId, React.ElementType> = {
  thinking: BrainCircuit,
  retrieving: Database,
  observing: Eye,
  responding: MessageSquare,
}

function buildStageGroups(steps: Message[]): StageGroup[] {
  const map = new Map<StageId, StageGroup>()
  const order: StageId[] = []

  for (const step of steps) {
    if (step.type !== 'thinking') continue
    const stage = (step.metadata?.stage as StageId | undefined) ?? 'thinking'
    const callState = step.metadata?.call_state ?? 'complete'
    const content = step.content ?? ''
    const ts = step.metadata?.timestamp ?? null

    if (!map.has(stage)) {
      order.push(stage)
      map.set(stage, {
        stage,
        label: STAGE_LABELS[stage] ?? stage,
        startedAt: ts,
        completedAt: callState === 'complete' ? ts : null,
        // start 事件 content 是 "xxx..." 标签文字，初始不显示
        content: callState === 'running' && !content.endsWith('...') ? content : '',
        state: callState === 'complete' ? 'complete' : 'running',
      })
    } else {
      const g = map.get(stage)!
      if (callState === 'complete') {
        g.state = 'complete'
        g.completedAt = ts
        // done 事件 content 为空，不覆盖已流式积累的内容
      } else {
        // running：content 是 ChatWindow 里累积的全量字符串，直接覆盖（非标签文字）
        if (g.startedAt === null) g.startedAt = ts
        if (content && !content.endsWith('...')) g.content = content
      }
    }
  }

  return order.map((s) => map.get(s)!)
}

function StageRow({
  group,
  isLast,
  isStreaming,
}: {
  group: StageGroup
  isLast: boolean
  isStreaming?: boolean
}) {
  const active = group.state === 'running' && isLast && Boolean(isStreaming)
  const Icon = STAGE_ICONS[group.stage] ?? BrainCircuit
  const [nowSec, setNowSec] = useState(() => Date.now() / 1000)
  // running 时默认展开，complete 时默认收起；用户点击可手动覆盖
  const [open, setOpen] = useState(group.state === 'running')
  const [userOverride, setUserOverride] = useState(false)
  const contentRef = useRef<HTMLDivElement | null>(null)

  // 阶段状态变化时同步 open（除非用户已手动改过）
  useEffect(() => {
    if (userOverride) return
    setOpen(group.state === 'running')
  }, [group.state, userOverride])

  // 流式内容更新时自动滚到底部
  useEffect(() => {
    if (open && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [group.content, open])

  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNowSec(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [active])

  let duration = ''
  if (group.completedAt !== null && group.startedAt !== null) {
    duration = `${Math.max(1, Math.round(group.completedAt - group.startedAt))}s`
  } else if (active && group.startedAt !== null) {
    duration = `${Math.max(1, Math.round(nowSec - group.startedAt))}s`
  }

  const hasContent = Boolean(group.content)

  const handleToggle = () => {
    if (!hasContent) return
    setUserOverride(true)
    setOpen((v) => !v)
  }

  const header = (
    <div
      className="flex items-center gap-2 py-0.5 text-[12px] font-medium text-slate-500 cursor-pointer hover:text-slate-700"
      onClick={handleToggle}
    >
      {hasContent
        ? <ChevronDown size={12} className={`shrink-0 transition-transform ${open ? '' : '-rotate-90'}`} />
        : <span className="w-3 shrink-0" />
      }
      <Icon size={12} strokeWidth={1.6} className="shrink-0" />
      <span>{group.label}{duration ? ` · ${duration}` : ''}</span>
      {active && <Loader2 size={11} className="animate-spin text-indigo-400" />}
    </div>
  )

  return (
    <div>
      {header}
      {hasContent && open && (
        <div
          ref={contentRef}
          className="ml-5 mr-3 mt-0.5 max-h-[180px] overflow-y-auto px-3 py-1 text-[11px] italic leading-relaxed text-slate-400 whitespace-pre-wrap"
        >
          {group.content}
        </div>
      )}
    </div>
  )
}

function toolLabel(tool: string): string {
  if (tool === 'solve_plan') return '解题计划'
  if (tool === 'solve_finish_step') return '完成步骤'
  if (tool === 'rag') return '检索知识库'
  if (tool === 'web_search') return '联网搜索'
  return tool
}

function ToolCallRow({ step }: { step: Message }) {
  const tool = String(step.metadata?.tool ?? 'tool')
  const input = step.metadata?.toolInput as Record<string, unknown> | undefined
  let summary = ''
  if (tool === 'solve_plan') {
    const arr = Array.isArray(input?.steps) ? (input!.steps as Array<{ goal?: string }>) : []
    summary = arr.map((st, i) => `${i + 1}. ${st.goal ?? ''}`).join('  ')
  } else if (tool === 'solve_finish_step') {
    summary = String(input?.summary ?? input?.step_id ?? '')
  } else if (tool === 'rag') {
    summary = String(input?.query ?? '')
  } else if (tool === 'web_search') {
    summary = String(input?.query ?? '')
  } else {
    summary = JSON.stringify(input ?? {}).slice(0, 80)
  }
  const label = toolLabel(tool)
  return (
    <div className="flex items-start gap-2 py-0.5 text-[12px] text-slate-500">
      <span className="shrink-0">🔧</span>
      <span className="font-medium shrink-0">{label}</span>
      {summary && <span className="text-slate-400 truncate">{summary}</span>}
    </div>
  )
}

function ToolCallGroup({ steps, isStreaming }: { steps: Message[]; isStreaming?: boolean }) {
  const labels = steps.map((s) => toolLabel(String(s.metadata?.tool ?? 'tool')))
  const allSame = labels.every((l) => l === labels[0])
  const groupLabel = allSame ? labels[0] : '工具调用'

  // 流式进行中默认展开（看实时进度），结束后默认收起；用户点击可手动覆盖（对齐 StageRow）
  const [open, setOpen] = useState(() => Boolean(isStreaming))
  const [userOverride, setUserOverride] = useState(false)
  useEffect(() => {
    if (userOverride) return
    setOpen(Boolean(isStreaming))
  }, [isStreaming, userOverride])

  const toggle = () => {
    setUserOverride(true)
    setOpen((v) => !v)
  }

  return (
    <div>
      <div
        className="flex items-center gap-2 py-0.5 text-[12px] font-medium text-slate-500 cursor-pointer hover:text-slate-700"
        onClick={toggle}
      >
        <ChevronDown size={12} className={`shrink-0 transition-transform ${open ? '' : '-rotate-90'}`} />
        <span className="shrink-0">🔧</span>
        <span>{groupLabel} · {steps.length} 次</span>
        {isStreaming && <Loader2 size={11} className="animate-spin text-indigo-400" />}
      </div>
      {open && (
        <div className="ml-5 mt-0.5 space-y-0.5">
          {steps.map((s, i) => (
            <ToolCallRow key={`tc-${i}`} step={s} />
          ))}
        </div>
      )}
    </div>
  )
}

export default function ThinkingProcess({ steps, isStreaming }: Props) {
  const groups = buildStageGroups(steps)
  const toolSteps = steps.filter((s) => s.type === 'tool_call')
  if (groups.length === 0 && toolSteps.length === 0) return null

  // 多轮工具调用（≥3）折叠成一组，避免撑爆聊天区；少量（≤2）则平铺，不过度设计
  const collapseTools = toolSteps.length >= 3

  return (
    <div className="mb-3 space-y-0.5 border-b border-slate-100 pb-3">
      {groups.map((g, i) => (
        <StageRow
          key={g.stage}
          group={g}
          isLast={i === groups.length - 1}
          isStreaming={isStreaming}
        />
      ))}
      {collapseTools ? (
        <ToolCallGroup steps={toolSteps} isStreaming={isStreaming} />
      ) : (
        toolSteps.map((s, i) => <ToolCallRow key={`tc-${i}`} step={s} />)
      )}
    </div>
  )
}
