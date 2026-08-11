import { useEffect, useState, useCallback } from 'react'
import { ArrowLeft, Network } from 'lucide-react'
import { fetchDashboard, type DashboardData } from '../../services/api'
import { getUser } from '../../services/auth'
import { Button, Skeleton } from '../ui'

// stale-while-revalidate：sessionStorage 缓存上次仪表盘数据，挂载先渲染旧值再后台刷新，
// 感知延迟接近零（止血方案，学情分析四模块设计 §模块四 P1）。按用户 id 命名空间，
// 避免共用浏览器时闪现他人数据。
const DASHBOARD_CACHE_PREFIX = 'dashboard:cache:'

function readDashboardCache(): DashboardData | null {
  const uid = getUser()?.id
  if (!uid) return null
  try {
    const raw = sessionStorage.getItem(DASHBOARD_CACHE_PREFIX + uid)
    return raw ? (JSON.parse(raw) as DashboardData) : null
  } catch {
    return null
  }
}

function writeDashboardCache(data: DashboardData): void {
  const uid = getUser()?.id
  if (!uid) return
  try {
    sessionStorage.setItem(DASHBOARD_CACHE_PREFIX + uid, JSON.stringify(data))
  } catch {
    /* quota / 隐私模式：忽略，缓存非必需 */
  }
}

export default function DashboardPanel({ onBack, onGraph }: { onBack: () => void; onGraph: () => void }) {
  // 挂载即尝试用缓存渲染（避免空白闪烁）；无缓存才显示 skeleton
  const initial = readDashboardCache()
  const [data, setData] = useState<DashboardData | null>(initial)
  const [loading, setLoading] = useState(initial === null)
  const [refreshing, setRefreshing] = useState(false)

  const load = useCallback(async () => {
    setRefreshing(true)
    try {
      const fresh = await fetchDashboard()
      setData(fresh)
      writeDashboardCache(fresh)
    } catch {
      /* 失败则保留已有数据（缓存或上次结果），不阻塞展示 */
    } finally {
      setRefreshing(false)
      setLoading(false)
    }
  }, [])

  // 挂载时后台拉取最新数据（load 是 useCallback 稳定引用，不会触发级联渲染）
  useEffect(() => { void load() }, [load])

  if (loading) {
    return <DashboardSkeleton onBack={onBack} onGraph={onGraph} />
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
        {refreshing && (
          <span className="text-xs text-muted animate-pulse">刷新中…</span>
        )}
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

/** 仪表盘骨架屏：镜像真实布局（4 统计卡 + 两段内容），消除全屏空白。 */
function DashboardSkeleton({ onBack, onGraph }: { onBack: () => void; onGraph: () => void }) {
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
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="rounded-[var(--radius)] border border-line bg-surface p-4">
                <Skeleton className="h-7 w-10" />
                <Skeleton className="h-3 w-14 mt-2" />
              </div>
            ))}
          </div>
          {[0, 1].map((s) => (
            <div key={s} className="bg-surface rounded-[var(--radius)] border border-line p-5 space-y-3">
              {[0, 1].map((i) => (
                <Skeleton key={i} className="h-5 w-3/4" />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
