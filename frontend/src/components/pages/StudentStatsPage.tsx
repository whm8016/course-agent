import { useEffect, useState, useCallback } from 'react'
import { authHeaders } from '../../services/auth'
import type { User } from '../../types'

interface Props {
  user: User
  onBack: () => void
}

/* ── Data types ────────────────────────────────────────────────────────────── */

interface CourseItem {
  course_id: string
  name: string
  status: string
  description?: string
  icon?: string
}

interface StudentSummary {
  student_id: string
  username: string
  display_name: string
  total_sessions: number
  total_messages: number
  total_questions: number
  correct_count: number
  accuracy_rate: number
  last_active_at: number | null
  knowledge_node_count: number
  high_risk_count: number
  error_node_count: number
  risk_score: number
}

interface HighRiskStudent {
  student_id: string
  display_name: string
  risk_score: number
  reasons: string[]
}

interface CourseStats {
  course_name: string
  total_students: number
  student_summaries: StudentSummary[]
  accuracy_distribution: Record<string, number>
  high_risk_students: HighRiskStudent[]
  daily_active_trend: { date: string; active_count: number }[]
}

interface StudentDetail {
  student: { id: string; username: string; display_name: string }
  sessions: number
  messages: number
  questions_total: number
  questions_correct: number
  accuracy_rate: number | null
  knowledge_graph: { nodes: KGNode[]; edges: unknown[] }
  error_graph: { nodes: ErrNode[]; edges: unknown[] }
  recent_questions: {
    question: string
    is_correct: boolean
    difficulty: string
    created_at: number
  }[]
}

interface KGNode {
  id: string
  label: string
  risk?: number
  mastery?: number
  importance?: number
  status?: string
  notes?: string
}

interface ErrNode {
  id: string
  label: string
  severity?: number
  error_count?: number
  correction_suggestions?: string[]
}

/* ── API helper ────────────────────────────────────────────────────────────── */

async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers || {}) },
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({})) as { detail?: string }
    throw new Error(data.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

/* ── Component ─────────────────────────────────────────────────────────────── */

export default function StudentStatsPage({ user, onBack }: Props) {
  const [courses, setCourses] = useState<CourseItem[]>([])
  const [selectedCourseId, setSelectedCourseId] = useState<string>('')
  const [stats, setStats] = useState<CourseStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sortBy, setSortBy] = useState<keyof StudentSummary>('risk_score')
  const [sortDesc, setSortDesc] = useState(true)

  // Student detail modal
  const [detailStudent, setDetailStudent] = useState<StudentDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  // ── Load courses ─────────────────────────────────────────────────────────
  const loadCourses = useCallback(async () => {
    try {
      const list = await apiFetch('/teacher/courses') as CourseItem[]
      setCourses(list)
      if (list.length > 0 && !selectedCourseId) {
        setSelectedCourseId(list[0].course_id)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载课程失败')
    }
  }, [selectedCourseId])

  useEffect(() => { void loadCourses() }, [loadCourses])

  // ── Load stats when course changes ───────────────────────────────────────
  const loadStats = useCallback(async (courseId: string) => {
    setLoading(true)
    setError('')
    setStats(null)
    try {
      const data = await apiFetch(`/teacher/courses/${courseId}/analytics/student-stats`) as CourseStats
      setStats(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载统计数据失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedCourseId) void loadStats(selectedCourseId)
  }, [selectedCourseId, loadStats])

  // ── Load student detail ──────────────────────────────────────────────────
  const openStudentDetail = async (studentId: string) => {
    setLoadingDetail(true)
    setDetailStudent(null)
    try {
      const data = await apiFetch(
        `/teacher/courses/${selectedCourseId}/analytics/student/${studentId}/detail`
      ) as StudentDetail
      setDetailStudent(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载学生详情失败')
    } finally {
      setLoadingDetail(false)
    }
  }

  // ── Sort logic ───────────────────────────────────────────────────────────
  const handleSort = (key: keyof StudentSummary) => {
    if (sortBy === key) {
      setSortDesc(prev => !prev)
    } else {
      setSortBy(key)
      setSortDesc(key === 'risk_score' || key === 'total_messages' || key === 'total_sessions')
    }
  }

  const sortedStudents = stats
    ? [...stats.student_summaries].sort((a, b) => {
        const av = a[sortBy] ?? 0
        const bv = b[sortBy] ?? 0
        const cmp = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv))
        return sortDesc ? -cmp : cmp
      })
    : []

  // ── Computed aggregates ──────────────────────────────────────────────────
  const avgAccuracy = stats && stats.student_summaries.length > 0
    ? stats.student_summaries.reduce((s, x) => s + x.accuracy_rate, 0) / stats.student_summaries.length
    : 0

  const highRiskCount = stats?.high_risk_students.length ?? 0

  // "today active" — students with last_active_at within today
  const todayStart = new Date()
  todayStart.setHours(0, 0, 0, 0)
  const todayTs = todayStart.getTime() / 1000
  const todayActive = stats
    ? stats.student_summaries.filter(s => s.last_active_at && s.last_active_at >= todayTs).length
    : 0

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 md:px-6 py-3 bg-white border-b border-slate-200 shrink-0">
        <button onClick={onBack} className="text-sm text-slate-500 hover:text-slate-800 transition">
          &larr; 返回
        </button>
        <h1 className="text-lg font-semibold text-slate-800">学生学情统计</h1>
        <span className="ml-auto text-xs text-slate-400">
          {user.display_name || user.username}
        </span>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: course list */}
        <aside className="w-60 border-r border-slate-200 bg-white overflow-y-auto shrink-0">
          <div className="p-3">
            <p className="text-xs font-medium text-slate-500 mb-2 px-1">选择课程</p>
            {courses.map(c => (
              <button
                key={c.course_id}
                onClick={() => setSelectedCourseId(c.course_id)}
                className={`w-full text-left px-3 py-2.5 rounded-lg mb-1 text-sm transition ${
                  selectedCourseId === c.course_id
                    ? 'bg-indigo-50 text-indigo-700 font-medium'
                    : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <span className="mr-1.5">{c.icon || '📘'}</span>
                {c.name}
              </button>
            ))}
          </div>
        </aside>

        {/* Right: stats panel */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          {loading && (
            <div className="flex items-center justify-center h-64 text-slate-400">加载中...</div>
          )}
          {error && (
            <div className="bg-red-50 text-red-600 rounded-lg p-4 text-sm">{error}</div>
          )}
          {!loading && !error && stats && (
            <div className="max-w-5xl mx-auto space-y-6">

              {/* ── 总览卡片 ────────────────────────────────────────────── */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="总学生" value={stats.total_students} color="bg-blue-50 text-blue-700" />
                <StatCard label="平均正确率" value={`${(avgAccuracy * 100).toFixed(0)}%`} color="bg-emerald-50 text-emerald-700" />
                <StatCard label="高风险学生" value={highRiskCount} color="bg-red-50 text-red-700" />
                <StatCard label="今日活跃" value={todayActive} color="bg-purple-50 text-purple-700" />
              </div>

              {/* ── 正确率分布 + 7 天活跃趋势 (并排) ───────────────────── */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* 正确率分布 */}
                <section className="bg-white rounded-2xl border border-slate-200 p-6">
                  <h3 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                    <span>📊</span> 答题正确率分布
                  </h3>
                  <AccuracyDistribution data={stats.accuracy_distribution} />
                </section>

                {/* 7 天活跃趋势 */}
                <section className="bg-white rounded-2xl border border-slate-200 p-6">
                  <h3 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                    <span>📈</span> 近 7 天活跃学生趋势
                  </h3>
                  {stats.daily_active_trend.length > 0 ? (
                    <div className="flex items-end gap-1 h-28">
                      {(() => {
                        const maxC = Math.max(...stats.daily_active_trend.map(d => d.active_count), 1)
                        return stats.daily_active_trend.map(d => (
                          <div key={d.date} className="flex-1 flex flex-col items-center gap-1">
                            <span className="text-[10px] text-slate-500">{d.active_count}</span>
                            <div
                              className="w-full bg-indigo-400 rounded-t"
                              style={{ height: `${(d.active_count / maxC) * 100}%`, minHeight: d.active_count > 0 ? 4 : 1 }}
                            />
                            <span className="text-[10px] text-slate-400">{d.date.slice(5)}</span>
                          </div>
                        ))
                      })()}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400">暂无数据</p>
                  )}
                </section>
              </div>

              {/* ── 高风险学生预警 ──────────────────────────────────────── */}
              {stats.high_risk_students.length > 0 && (
                <section className="bg-white rounded-2xl border border-red-200 p-6">
                  <h3 className="font-semibold text-red-700 mb-4 flex items-center gap-2">
                    <span>⚠️</span> 高风险学生预警
                    <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full">
                      {stats.high_risk_students.length} 人
                    </span>
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {stats.high_risk_students.map(s => (
                      <div key={s.student_id} className="flex items-start gap-3 p-3 rounded-lg bg-red-50 border border-red-100">
                        <div className="flex-1">
                          <p className="text-sm font-medium text-slate-800">{s.display_name}</p>
                          <p className="text-xs text-slate-500 mt-0.5">{s.reasons.join('、')}</p>
                        </div>
                        <span className="text-xs font-bold text-red-600 whitespace-nowrap">
                          风险 {(s.risk_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* ── 学生列表表格 ────────────────────────────────────────── */}
              <section className="bg-white rounded-2xl border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                  <span>👥</span> 学生学情一览
                </h3>
                {sortedStudents.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-100 text-slate-500 text-xs">
                          <th className="text-left py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('display_name')}>
                            姓名 {sortBy === 'display_name' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('total_sessions')}>
                            会话 {sortBy === 'total_sessions' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('total_messages')}>
                            提问 {sortBy === 'total_messages' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('total_questions')}>
                            答题 {sortBy === 'total_questions' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('accuracy_rate')}>
                            正确率 {sortBy === 'accuracy_rate' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-center py-2 px-2 cursor-pointer hover:text-slate-700" onClick={() => handleSort('risk_score')}>
                            风险 {sortBy === 'risk_score' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2">最后活跃</th>
                          <th className="text-center py-2 px-2">详情</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-50">
                        {sortedStudents.map(s => (
                          <tr key={s.student_id} className="hover:bg-slate-50 transition">
                            <td className="py-2.5 px-2 font-medium text-slate-800">
                              {s.display_name || s.username}
                            </td>
                            <td className="py-2.5 px-2 text-right text-slate-600">{s.total_sessions}</td>
                            <td className="py-2.5 px-2 text-right text-slate-600">{s.total_messages}</td>
                            <td className="py-2.5 px-2 text-right text-slate-600">{s.total_questions}</td>
                            <td className="py-2.5 px-2 text-right">
                              {s.total_questions > 0
                                ? <span className={s.accuracy_rate >= 0.6 ? 'text-emerald-600' : s.accuracy_rate >= 0.4 ? 'text-amber-600' : 'text-red-600'}>
                                    {(s.accuracy_rate * 100).toFixed(0)}%
                                  </span>
                                : <span className="text-slate-400">--</span>
                              }
                            </td>
                            <td className="py-2.5 px-2 text-center">
                              <RiskBadge score={s.risk_score} />
                            </td>
                            <td className="py-2.5 px-2 text-right text-xs text-slate-400">
                              {s.last_active_at
                                ? new Date(s.last_active_at * 1000).toLocaleDateString()
                                : '--'
                              }
                            </td>
                            <td className="py-2.5 px-2 text-center">
                              <button
                                onClick={() => void openStudentDetail(s.student_id)}
                                className="text-xs text-indigo-600 hover:text-indigo-800 transition"
                              >
                                查看
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400 text-center py-8">暂无学生数据</p>
                )}
              </section>
            </div>
          )}
          {!loading && !error && !stats && selectedCourseId && (
            <div className="flex items-center justify-center h-64 text-slate-400">加载中...</div>
          )}
          {!selectedCourseId && courses.length === 0 && (
            <div className="flex items-center justify-center h-64 text-slate-400">暂无课程</div>
          )}
        </main>
      </div>

      {/* ── 学生详情弹窗 ────────────────────────────────────────────── */}
      {(detailStudent || loadingDetail) && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={() => setDetailStudent(null)}>
          <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            {loadingDetail ? (
              <div className="flex items-center justify-center py-16 text-slate-400">加载中...</div>
            ) : detailStudent ? (
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold text-slate-800">
                    {detailStudent.student.display_name || detailStudent.student.username} 的学习详情
                  </h2>
                  <button onClick={() => setDetailStudent(null)} className="text-slate-400 hover:text-slate-600 text-xl">&times;</button>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-4 gap-3 mb-6">
                  <div className="rounded-xl p-3 bg-blue-50 text-blue-700 text-center">
                    <p className="text-lg font-bold">{detailStudent.sessions}</p>
                    <p className="text-xs opacity-70">会话数</p>
                  </div>
                  <div className="rounded-xl p-3 bg-emerald-50 text-emerald-700 text-center">
                    <p className="text-lg font-bold">{detailStudent.messages}</p>
                    <p className="text-xs opacity-70">提问数</p>
                  </div>
                  <div className="rounded-xl p-3 bg-amber-50 text-amber-700 text-center">
                    <p className="text-lg font-bold">{detailStudent.questions_total}</p>
                    <p className="text-xs opacity-70">答题数</p>
                  </div>
                  <div className="rounded-xl p-3 bg-purple-50 text-purple-700 text-center">
                    <p className="text-lg font-bold">
                      {detailStudent.accuracy_rate != null ? `${(detailStudent.accuracy_rate * 100).toFixed(0)}%` : '--'}
                    </p>
                    <p className="text-xs opacity-70">正确率</p>
                  </div>
                </div>

                {/* Knowledge graph summary */}
                <section className="mb-6">
                  <h3 className="text-sm font-semibold text-slate-700 mb-2">知识点掌握情况</h3>
                  <div className="grid grid-cols-3 gap-3 mb-3">
                    <div className="rounded-lg p-2 bg-slate-50 text-center">
                      <p className="text-base font-bold text-slate-700">{detailStudent.knowledge_graph.nodes.length}</p>
                      <p className="text-[11px] text-slate-500">知识点总数</p>
                    </div>
                    <div className="rounded-lg p-2 bg-red-50 text-center">
                      <p className="text-base font-bold text-red-600">
                        {detailStudent.knowledge_graph.nodes.filter(n => (n.risk ?? 0) > 0.6).length}
                      </p>
                      <p className="text-[11px] text-red-500">高风险点</p>
                    </div>
                    <div className="rounded-lg p-2 bg-amber-50 text-center">
                      <p className="text-base font-bold text-amber-600">{detailStudent.error_graph.nodes.length}</p>
                      <p className="text-[11px] text-amber-500">错误模式</p>
                    </div>
                  </div>
                  {/* High risk knowledge points */}
                  {detailStudent.knowledge_graph.nodes.filter(n => (n.risk ?? 0) > 0.6).length > 0 && (
                    <div className="space-y-2">
                      {detailStudent.knowledge_graph.nodes
                        .filter(n => (n.risk ?? 0) > 0.6)
                        .sort((a, b) => (b.risk ?? 0) - (a.risk ?? 0))
                        .slice(0, 5)
                        .map(n => (
                          <div key={n.id} className="flex items-center gap-3 text-sm">
                            <span className="flex-1 text-slate-700">{n.label}</span>
                            <span className="text-xs text-red-600">风险 {((n.risk ?? 0) * 100).toFixed(0)}%</span>
                            <span className="text-xs text-emerald-600">掌握 {((n.mastery ?? 0) * 100).toFixed(0)}%</span>
                            <div className="w-16 h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div className="h-full bg-red-500 rounded-full" style={{ width: `${((n.risk ?? 0) * 100)}%` }} />
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                </section>

                {/* Recent questions */}
                {detailStudent.recent_questions.length > 0 && (
                  <section>
                    <h3 className="text-sm font-semibold text-slate-700 mb-2">近期答题记录</h3>
                    <div className="divide-y divide-slate-100">
                      {detailStudent.recent_questions.map((q, i) => (
                        <div key={i} className="flex items-center gap-3 py-2.5">
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            q.is_correct ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-600'
                          }`}>
                            {q.is_correct ? '✓' : '✗'}
                          </span>
                          <span className="flex-1 text-sm text-slate-700 truncate" title={q.question}>
                            {q.question}
                          </span>
                          {q.difficulty && (
                            <span className="text-xs text-slate-400">{q.difficulty}</span>
                          )}
                          <span className="text-[11px] text-slate-400">
                            {new Date(q.created_at * 1000).toLocaleDateString()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Sub-components ────────────────────────────────────────────────────────── */

function StatCard({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className={`rounded-xl p-4 ${color}`}>
      <p className="text-xs opacity-70 mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  )
}

function RiskBadge({ score }: { score: number }) {
  if (score > 0.6) {
    return <span className="text-xs font-medium bg-red-100 text-red-700 px-2 py-0.5 rounded-full">高风险</span>
  }
  if (score > 0.35) {
    return <span className="text-xs font-medium bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">中等</span>
  }
  return <span className="text-xs font-medium bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full">良好</span>
}

function AccuracyDistribution({ data }: { data: Record<string, number> }) {
  const buckets = [
    { key: '0_20', label: '0-20%', color: 'bg-red-400' },
    { key: '20_40', label: '20-40%', color: 'bg-orange-400' },
    { key: '40_60', label: '40-60%', color: 'bg-amber-400' },
    { key: '60_80', label: '60-80%', color: 'bg-emerald-400' },
    { key: '80_100', label: '80-100%', color: 'bg-teal-500' },
  ]
  const maxVal = Math.max(...buckets.map(b => data[b.key] ?? 0), 1)

  return (
    <div className="flex items-end gap-2 h-28">
      {buckets.map(b => {
        const val = data[b.key] ?? 0
        return (
          <div key={b.key} className="flex-1 flex flex-col items-center gap-1">
            <span className="text-[10px] text-slate-500">{val}</span>
            <div
              className={`w-full ${b.color} rounded-t`}
              style={{ height: `${(val / maxVal) * 100}%`, minHeight: val > 0 ? 4 : 1 }}
            />
            <span className="text-[10px] text-slate-400">{b.label}</span>
          </div>
        )
      })}
    </div>
  )
}
