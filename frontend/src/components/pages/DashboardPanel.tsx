import { useEffect, useState, useCallback } from 'react'
import { ArrowLeft, Network } from 'lucide-react'
import { fetchDashboard, type DashboardData } from '../../services/api'
import { Button } from '../ui'

export default function DashboardPanel({ onBack, onGraph }: { onBack: () => void; onGraph: () => void }) {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setData(await fetchDashboard())
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  // 挂载时拉取一次数据（load 是 useCallback 稳定引用，不会触发级联渲染）
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void load() }, [load])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-canvas text-muted">
        加载中...
      </div>
    )
  }

  if (!data) {
    return (
      <div className="h-screen flex items-center justify-center bg-canvas text-muted">
        暂无数据
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-canvas">
      <header className="flex items-center gap-4 px-4 md:px-6 py-3 bg-surface border-b border-line">
        <Button variant="ghost" size="sm" icon={ArrowLeft} onClick={onBack}>
          返回
        </Button>
        <h1 className="text-lg font-serif text-ink">学习仪表盘</h1>
        <div className="ml-auto">
          <Button variant="primary" size="sm" icon={Network} onClick={onGraph}>
            查看图谱
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* 统计卡片 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="知识点" value={data.knowledge_node_count} tone="info" />
            <StatCard label="错误模式" value={data.error_node_count} tone="danger" />
            <StatCard label="高风险点" value={data.high_risk_points.length} tone="warn" />
            <StatCard label="高频错误" value={data.frequent_errors.length} tone="neutral" />
          </div>

          {/* 高风险知识点 */}
          {data.high_risk_points.length > 0 && (
            <section className="bg-surface rounded-[var(--radius)] border border-line p-5">
              <h2 className="text-base font-serif text-warn-fg mb-3">高风险知识点</h2>
              <div className="space-y-3">
                {data.high_risk_points.map((node) => (
                  <div key={node.id} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-ink">{node.label}</div>
                      {node.notes && <div className="text-xs text-muted mt-0.5">{node.notes}</div>}
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-danger-fg">风险 {((node.risk ?? 0) * 100).toFixed(0)}%</span>
                      <span className="text-ok-fg">掌握 {((node.mastery ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                    <RiskBar value={node.risk ?? 0} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 高频错误 */}
          {data.frequent_errors.length > 0 && (
            <section className="bg-surface rounded-[var(--radius)] border border-line p-5">
              <h2 className="text-base font-serif text-danger-fg mb-3">高频错误</h2>
              <div className="space-y-3">
                {data.frequent_errors.map((node) => (
                  <div key={node.id} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-ink">{node.label}</div>
                      {node.notes && <div className="text-xs text-muted mt-0.5">{node.notes}</div>}
                    </div>
                    <span className="text-xs text-ink-soft">出错 {node.error_count ?? 1} 次</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 学习摘要 */}
          {data.summary && (
            <section className="bg-surface rounded-[var(--radius)] border border-line p-5">
              <h2 className="text-base font-serif text-ink mb-3">学习轨迹</h2>
              <div className="text-sm text-ink-soft whitespace-pre-wrap leading-relaxed">
                {data.summary.slice(-2000)}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

type StatTone = 'info' | 'danger' | 'warn' | 'neutral'

function StatCard({ label, value, tone }: { label: string; value: number; tone: StatTone }) {
  const toneMap: Record<StatTone, string> = {
    info: 'bg-info-bg text-info-fg border-info-bg',
    danger: 'bg-danger-bg text-danger-fg border-danger-bg',
    warn: 'bg-warn-bg text-warn-fg border-warn-bg',
    neutral: 'bg-neutral-bg text-neutral-fg border-neutral-bg',
  }
  return (
    <div className={`rounded-[var(--radius)] border p-4 ${toneMap[tone]}`}>
      <div className="text-2xl font-serif">{value}</div>
      <div className="text-xs mt-1 opacity-80">{label}</div>
    </div>
  )
}

function RiskBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="w-16 h-2 bg-surface-2 rounded-full overflow-hidden">
      <div
        className="h-full bg-danger-fg rounded-full"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
