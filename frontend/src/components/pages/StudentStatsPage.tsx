import { useEffect, useState, useCallback } from 'react'
import { BarChart3, TrendingUp, AlertTriangle, Users } from 'lucide-react'
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
    <div className="h-screen flex flex-col bg-canvas">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 md:px-6 py-3 bg-surface border-b border-line shrink-0">
        <button onClick={onBack} className="text-sm text-ink-soft hover:text-ink transition">
          &larr; 返回
        </button>
        <h1 className="text-lg font-serif text-ink">学生学情统计</h1>
        <span className="ml-auto text-xs text-muted">
          {user.display_name || user.username}
        </span>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: course list */}
        <aside className="w-60 border-r border-line bg-surface overflow-y-auto shrink-0">
          <div className="p-3">
            <p className="text-xs font-medium text-muted mb-2 px-1">选择课程</p>
            {courses.map(c => (
              <button
                key={c.course_id}
                onClick={() => setSelectedCourseId(c.course_id)}
                className={`w-full text-left px-3 py-2.5 rounded-[var(--radius)] mb-1 text-sm transition ${
                  selectedCourseId === c.course_id
                    ? 'bg-surface-2 text-ink font-medium'
                    : 'text-ink-soft hover:bg-surface-2'
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
            <div className="flex items-center justify-center h-64 text-muted">加载中...</div>
          )}
          {error && (
            <div className="bg-danger-bg text-danger-fg rounded-[var(--radius)] p-4 text-sm">{error}</div>
          )}
          {!loading && !error && stats && (
            <div className="max-w-5xl mx-auto space-y-6">

              {/* ── 总览卡片 ────────────────────────────────────────────── */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="总学生" value={stats.total_students} tone="info" />
                <StatCard label="平均正确率" value={`${(avgAccuracy * 100).toFixed(0)}%`} tone="ok" />
                <StatCard label="高风险学生" value={highRiskCount} tone="danger" />
                <StatCard label="今日活跃" value={todayActive} tone="neutral" />
              </div>

              {/* ── 正确率分布 + 7 天活跃趋势 (并排) ───────────────────── */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* 正确率分布 */}
                <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                  <h3 className="font-serif text-ink mb-4 flex items-center gap-2">
                    <BarChart3 size={16} strokeWidth={1.5} className="text-ink-soft" />
                    答题正确率分布
                  </h3>
                  <AccuracyDistribution data={stats.accuracy_distribution} />
                </section>

                {/* 7 天活跃趋势 */}
                <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                  <h3 className="font-serif text-ink mb-4 flex items-center gap-2">
                    <TrendingUp size={16} strokeWidth={1.5} className="text-ink-soft" />
                    近 7 天活跃学生趋势
                  </h3>
                  {stats.daily_active_trend.length > 0 ? (
                    <div className="flex items-end gap-1 h-28">
                      {(() => {
                        const maxC = Math.max(...stats.daily_active_trend.map(d => d.active_count), 1)
                        return stats.daily_active_trend.map(d => (
                          <div key={d.date} className="flex-1 flex flex-col items-center gap-1">
                            <span className="text-[10px] text-muted">{d.active_count}</span>
                            <div
                              className="w-full bg-info-fg rounded-t-[var(--radius-sm)]"
                              style={{ height: `${(d.active_count / maxC) * 100}%`, minHeight: d.active_count > 0 ? 4 : 1 }}
                            />
                            <span className="text-[10px] text-muted">{d.date.slice(5)}</span>
                          </div>
                        ))
                      })()}
                    </div>
                  ) : (
                    <p className="text-sm text-muted">暂无数据</p>
                  )}
                </section>
              </div>

              {/* ── 高风险学生预警 ──────────────────────────────────────── */}
              {stats.high_risk_students.length > 0 && (
                <section className="bg-surface rounded-[var(--radius-lg)] border border-danger-bg p-6">
                  <h3 className="font-serif text-danger-fg mb-4 flex items-center gap-2">
                    <AlertTriangle size={16} strokeWidth={1.5} />
                    高风险学生预警
                    <span className="text-xs bg-danger-bg text-danger-fg px-2 py-0.5 rounded-full font-sans">
                      {stats.high_risk_students.length} 人
                    </span>
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {stats.high_risk_students.map(s => (
                      <div key={s.student_id} className="flex items-start gap-3 p-3 rounded-[var(--radius)] bg-danger-bg border border-line">
                        <div className="flex-1">
                          <p className="text-sm font-medium text-ink">{s.display_name}</p>
                          <p className="text-xs text-ink-soft mt-0.5">{s.reasons.join('、')}</p>
                        </div>
                        <span className="text-xs font-bold text-danger-fg whitespace-nowrap">
                          风险 {(s.risk_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* ── 学生列表表格 ────────────────────────────────────────── */}
              <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                <h3 className="font-serif text-ink mb-4 flex items-center gap-2">
                  <Users size={16} strokeWidth={1.5} className="text-ink-soft" />
                  学生学情一览
                </h3>
                {sortedStudents.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-line text-muted text-xs">
                          <th className="text-left py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('display_name')}>
                            姓名 {sortBy === 'display_name' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('total_sessions')}>
                            会话 {sortBy === 'total_sessions' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('total_messages')}>
                            提问 {sortBy === 'total_messages' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('total_questions')}>
                            答题 {sortBy === 'total_questions' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('accuracy_rate')}>
                            正确率 {sortBy === 'accuracy_rate' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-center py-2 px-2 cursor-pointer hover:text-ink" onClick={() => handleSort('risk_score')}>
                            风险 {sortBy === 'risk_score' && (sortDesc ? '↓' : '↑')}
                          </th>
                          <th className="text-right py-2 px-2">最后活跃</th>
                          <th className="text-center py-2 px-2">详情</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line">
                        {sortedStudents.map(s => (
                          <tr key={s.student_id} className="hover:bg-surface-2 transition">
                            <td className="py-2.5 px-2 font-medium text-ink">
                              {s.display_name || s.username}
                            </td>
                            <td className="py-2.5 px-2 text-right text-ink-soft">{s.total_sessions}</td>
                            <td className="py-2.5 px-2 text-right text-ink-soft">{s.total_messages}</td>
                            <td className="py-2.5 px-2 text-right text-ink-soft">{s.total_questions}</td>
                            <td className="py-2.5 px-2 text-right">
                              {s.total_questions > 0
                                ? <span className={s.accuracy_rate >= 0.6 ? 'text-ok-fg' : s.accuracy_rate >= 0.4 ? 'text-warn-fg' : 'text-danger-fg'}>
                                    {(s.accuracy_rate * 100).toFixed(0)}%
                                  </span>
                                : <span className="text-muted">--</span>
                              }
                            </td>
                            <td className="py-2.5 px-2 text-center">
                              <RiskBadge score={s.risk_score} />
                            </td>
                            <td className="py-2.5 px-2 text-right text-xs text-muted">
                              {s.last_active_at
                                ? new Date(s.last_active_at * 1000).toLocaleDateString()
                                : '--'
                              }
                            </td>
                            <td className="py-2.5 px-2 text-center">
                              <button
                                onClick={() => void openStudentDetail(s.student_id)}
                                className="text-xs text-ink hover:text-accent underline-offset-2 hover:underline transition"
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
                  <p className="text-sm text-muted text-center py-8">暂无学生数据</p>
                )}
              </section>
            </div>
          )}
          {!loading && !error && !stats && selectedCourseId && (
            <div className="flex items-center justify-center h-64 text-muted">加载中...</div>
          )}
          {!selectedCourseId && courses.length === 0 && (
            <div className="flex items-center justify-center h-64 text-muted">暂无课程</div>
          )}
        </main>
      </div>

      {/* ── 学生详情弹窗 ────────────────────────────────────────────── */}
      {(detailStudent || loadingDetail) && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={() => setDetailStudent(null)}>
          <div className="bg-surface rounded-[var(--radius-lg)] w-full max-w-2xl max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            {loadingDetail ? (
              <div className="flex items-center justify-center py-16 text-muted">加载中...</div>
            ) : detailStudent ? (
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-serif text-ink">
                    {detailStudent.student.display_name || detailStudent.student.username} 的学习详情
                  </h2>
                  <button onClick={() => setDetailStudent(null)} className="text-muted hover:text-ink text-xl">&times;</button>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-4 gap-3 mb-6">
                  <div className="rounded-[var(--radius)] p-3 bg-info-bg text-info-fg text-center">
                    <p className="text-lg font-serif">{detailStudent.sessions}</p>
                    <p className="text-xs opacity-80">会话数</p>
                  </div>
                  <div className="rounded-[var(--radius)] p-3 bg-ok-bg text-ok-fg text-center">
                    <p className="text-lg font-serif">{detailStudent.messages}</p>
                    <p className="text-xs opacity-80">提问数</p>
                  </div>
                  <div className="rounded-[var(--radius)] p-3 bg-warn-bg text-warn-fg text-center">
                    <p className="text-lg font-serif">{detailStudent.questions_total}</p>
                    <p className="text-xs opacity-80">答题数</p>
                  </div>
                  <div className="rounded-[var(--radius)] p-3 bg-neutral-bg text-neutral-fg text-center">
                    <p className="text-lg font-serif">
                      {detailStudent.accuracy_rate != null ? `${(detailStudent.accuracy_rate * 100).toFixed(0)}%` : '--'}
                    </p>
                    <p className="text-xs opacity-80">正确率</p>
                  </div>
                </div>

                {/* Knowledge graph summary */}
                <section className="mb-6">
                  <h3 className="text-sm font-semibold text-ink mb-2">知识点掌握情况</h3>
                  <div className="grid grid-cols-3 gap-3 mb-3">
                    <div className="rounded-[var(--radius)] p-2 bg-surface-2 text-center">
                      <p className="text-base font-serif text-ink">{detailStudent.knowledge_graph.nodes.length}</p>
                      <p className="text-[11px] text-muted">知识点总数</p>
                    </div>
                    <div className="rounded-[var(--radius)] p-2 bg-danger-bg text-center">
                      <p className="text-base font-serif text-danger-fg">
                        {detailStudent.knowledge_graph.nodes.filter(n => (n.risk ?? 0) > 0.6).length}
                      </p>
                      <p className="text-[11px] text-danger-fg">高风险点</p>
                    </div>
                    <div className="rounded-[var(--radius)] p-2 bg-warn-bg text-center">
                      <p className="text-base font-serif text-warn-fg">{detailStudent.error_graph.nodes.length}</p>
                      <p className="text-[11px] text-warn-fg">错误模式</p>
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
                            <span className="flex-1 text-ink-soft">{n.label}</span>
                            <span className="text-xs text-danger-fg">风险 {((n.risk ?? 0) * 100).toFixed(0)}%</span>
                            <span className="text-xs text-ok-fg">掌握 {((n.mastery ?? 0) * 100).toFixed(0)}%</span>
                            <div className="w-16 h-2 bg-surface-2 rounded-full overflow-hidden">
                              <div className="h-full bg-danger-fg rounded-full" style={{ width: `${((n.risk ?? 0) * 100)}%` }} />
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                </section>

                {/* Recent questions */}
                {detailStudent.recent_questions.length > 0 && (
                  <section>
                    <h3 className="text-sm font-semibold text-ink mb-2">近期答题记录</h3>
                    <div className="divide-y divide-line">
                      {detailStudent.recent_questions.map((q, i) => (
                        <div key={i} className="flex items-center gap-3 py-2.5">
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            q.is_correct ? 'bg-ok-bg text-ok-fg' : 'bg-danger-bg text-danger-fg'
                          }`}>
                            {q.is_correct ? '✓' : '✗'}
                          </span>
                          <span className="flex-1 text-sm text-ink-soft truncate" title={q.question}>
                            {q.question}
                          </span>
                          {q.difficulty && (
                            <span className="text-xs text-muted">{q.difficulty}</span>
                          )}
                          <span className="text-[11px] text-muted">
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

type Tone = 'info' | 'ok' | 'danger' | 'warn' | 'neutral'

function StatCard({ label, value, tone }: { label: string; value: number | string; tone: Tone }) {
  const toneMap: Record<Tone, string> = {
    info: 'bg-info-bg text-info-fg',
    ok: 'bg-ok-bg text-ok-fg',
    danger: 'bg-danger-bg text-danger-fg',
    warn: 'bg-warn-bg text-warn-fg',
    neutral: 'bg-neutral-bg text-neutral-fg',
  }
  return (
    <div className={`rounded-[var(--radius)] p-4 ${toneMap[tone]}`}>
      <p className="text-xs opacity-80 mb-1">{label}</p>
      <p className="text-2xl font-serif">{value}</p>
    </div>
  )
}

function RiskBadge({ score }: { score: number }) {
  if (score > 0.6) {
    return <span className="text-xs font-medium bg-danger-bg text-danger-fg px-2 py-0.5 rounded-full">高风险</span>
  }
  if (score > 0.35) {
    return <span className="text-xs font-medium bg-warn-bg text-warn-fg px-2 py-0.5 rounded-full">中等</span>
  }
  return <span className="text-xs font-medium bg-ok-bg text-ok-fg px-2 py-0.5 rounded-full">良好</span>
}

function AccuracyDistribution({ data }: { data: Record<string, number> }) {
  const buckets = [
    { key: '0_20', label: '0-20%', color: 'bg-danger-fg' },
    { key: '20_40', label: '20-40%', color: 'bg-warn-fg' },
    { key: '40_60', label: '40-60%', color: 'bg-neutral-fg' },
    { key: '60_80', label: '60-80%', color: 'bg-ok-fg' },
    { key: '80_100', label: '80-100%', color: 'bg-info-fg' },
  ]
  const maxVal = Math.max(...buckets.map(b => data[b.key] ?? 0), 1)

  return (
    <div className="flex items-end gap-2 h-28">
      {buckets.map(b => {
        const val = data[b.key] ?? 0
        return (
          <div key={b.key} className="flex-1 flex flex-col items-center gap-1">
            <span className="text-[10px] text-muted">{val}</span>
            <div
              className={`w-full ${b.color} rounded-t-[var(--radius-sm)]`}
              style={{ height: `${(val / maxVal) * 100}%`, minHeight: val > 0 ? 4 : 1 }}
            />
            <span className="text-[10px] text-muted">{b.label}</span>
          </div>
        )
      })}
    </div>
  )
}
