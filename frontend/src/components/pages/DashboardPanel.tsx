import { useEffect, useState, useCallback } from 'react'
import { fetchDashboard, type DashboardData } from '../../services/api'

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

  useEffect(() => { void load() }, [load])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50 text-slate-400">
        加载中...
      </div>
    )
  }

  if (!data) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50 text-slate-400">
        暂无数据
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <header className="flex items-center gap-4 px-4 md:px-6 py-3 bg-white border-b border-slate-200">
        <button onClick={onBack} className="text-sm text-slate-500 hover:text-slate-800">&larr; 返回</button>
        <h1 className="text-lg font-semibold">学习仪表盘</h1>
        <button
          onClick={onGraph}
          className="ml-auto px-3 py-1 rounded text-sm bg-blue-600 text-white hover:bg-blue-700"
        >查看图谱</button>
      </header>

      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* 统计卡片 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="知识点" value={data.knowledge_node_count} color="blue" />
            <StatCard label="错误模式" value={data.error_node_count} color="red" />
            <StatCard label="高风险点" value={data.high_risk_points.length} color="amber" />
            <StatCard label="高频错误" value={data.frequent_errors.length} color="rose" />
          </div>

          {/* 高风险知识点 */}
          {data.high_risk_points.length > 0 && (
            <section className="bg-white rounded-lg border border-slate-200 p-5">
              <h2 className="text-base font-semibold mb-3 text-amber-700">高风险知识点</h2>
              <div className="space-y-3">
                {data.high_risk_points.map((node) => (
                  <div key={node.id} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="text-sm font-medium">{node.label}</div>
                      {node.notes && <div className="text-xs text-slate-500 mt-0.5">{node.notes}</div>}
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-red-600">风险 {((node.risk ?? 0) * 100).toFixed(0)}%</span>
                      <span className="text-green-600">掌握 {((node.mastery ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                    <RiskBar value={node.risk ?? 0} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 高频错误 */}
          {data.frequent_errors.length > 0 && (
            <section className="bg-white rounded-lg border border-slate-200 p-5">
              <h2 className="text-base font-semibold mb-3 text-rose-700">高频错误</h2>
              <div className="space-y-3">
                {data.frequent_errors.map((node) => (
                  <div key={node.id} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="text-sm font-medium">{node.label}</div>
                      {node.notes && <div className="text-xs text-slate-500 mt-0.5">{node.notes}</div>}
                    </div>
                    <span className="text-xs text-slate-600">出错 {node.error_count ?? 1} 次</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 学习摘要 */}
          {data.summary && (
            <section className="bg-white rounded-lg border border-slate-200 p-5">
              <h2 className="text-base font-semibold mb-3">学习轨迹</h2>
              <div className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                {data.summary.slice(-2000)}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    rose: 'bg-rose-50 text-rose-700 border-rose-200',
  }
  return (
    <div className={`rounded-lg border p-4 ${colorMap[color] || colorMap.blue}`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs mt-1 opacity-80">{label}</div>
    </div>
  )
}

function RiskBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="w-16 h-2 bg-slate-100 rounded-full overflow-hidden">
      <div
        className="h-full bg-red-500 rounded-full"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
