import type { UsageSummary } from '../../services/api'

// 维度名 → 中文标签（成本排行表头用）
const DIM_LABEL: Record<string, string> = {
  course: '课程',
  user: '用户',
  model: '模型',
  day: '日期',
}

function fmtTokens(n: number): string {
  // 大数压缩成 1.2k，小数原样，便于在窄列里读
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k'
  return String(n)
}

interface Props {
  usage: UsageSummary
  dim: 'course' | 'user' | 'model' | 'day'
}

/** LLM 用量展示：汇总卡片（输入/输出/缓存命中/成本）+ 按维度成本排行条 + 明细。
 *
 * 纯 CSS div 条形（width:cost/maxCost%），不引图表库，视觉与 TeacherPage 活跃度趋势一致。
 * admin（全量，可切维度）与 teacher（单课程，默认按用户）共用。
 */
export default function UsagePanel({ usage, dim }: Props) {
  const t = usage.total
  const cacheHit = t.input_tokens > 0 ? Math.round((t.cache_read_tokens / t.input_tokens) * 100) : null
  const dimLabel = DIM_LABEL[dim] ?? dim
  const dimKey = dim
  const maxCost = Math.max(...usage.rows.map(r => r.cost_usd), 0.0001)

  return (
    <>
      {/* 汇总卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: '输入 tokens', value: fmtTokens(t.input_tokens), color: 'bg-info-bg text-info-fg' },
          { label: '输出 tokens', value: fmtTokens(t.output_tokens), color: 'bg-ok-bg text-ok-fg' },
          { label: '缓存命中', value: cacheHit === null ? '—' : cacheHit + '%', color: 'bg-warn-bg text-warn-fg' },
          { label: '总成本', value: '$' + t.cost_usd.toFixed(4), color: 'bg-info-bg text-info-fg' },
        ].map(card => (
          <div key={card.label} className={`rounded-[var(--radius)] p-4 ${card.color}`}>
            <p className="text-xs opacity-70 mb-1">{card.label}</p>
            <p className="text-2xl font-bold">{card.value}</p>
          </div>
        ))}
      </div>

      {/* 成本排行条 + 明细 */}
      <div className="bg-surface rounded-[var(--radius)] border border-line overflow-hidden">
        <div className="px-4 py-3 border-b border-line text-xs text-ink-soft">
          按{dimLabel}成本排行（共 {usage.rows.length} 项，按 cost 降序）
        </div>
        <div className="divide-y divide-line">
          {usage.rows.map((r, i) => {
            const raw = (r[dimKey] ?? '—') as string
            // day 维度的 "YYYYMMDD" 压成 "MM-DD" 便于读
            const label = dim === 'day' && /^\d{8}$/.test(raw) ? `${raw.slice(4, 6)}-${raw.slice(6, 8)}` : raw
            return (
              <div key={i} className="px-4 py-2.5 flex items-center gap-3">
                <span className="w-40 shrink-0 truncate text-sm text-ink" title={raw}>{label}</span>
                <div className="flex-1 bg-canvas rounded h-4 overflow-hidden">
                  <div
                    className="h-full bg-accent rounded"
                    style={{ width: `${(r.cost_usd / maxCost) * 100}%`, minWidth: 2 }}
                  />
                </div>
                <span className="w-24 shrink-0 text-right text-sm tabular-nums text-ink-soft">
                  ${r.cost_usd.toFixed(4)}
                </span>
                <span className="w-36 shrink-0 text-right text-xs text-muted tabular-nums">
                  输{fmtTokens(r.input_tokens)} · 出{fmtTokens(r.output_tokens)} · {r.call_count}次
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
