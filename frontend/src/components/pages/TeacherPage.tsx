import { useEffect, useRef, useState } from 'react'
import { authHeaders } from '../../services/auth'
import type { User } from '../../types'
import JoinCodeShareSection from './JoinCodeShareSection'
import KbDetailPanel from './KbDetailPanel'
import { STATUS_LABEL, STATUS_COLOR } from './kbUtils'
import type { KB } from './KbDetailPanel'

interface Props {
  user: User
  onBack: () => void
}

interface Student {
  id: string
  username: string
  display_name: string
  enrolled_at: number
}

interface OverviewData {
  total_students: number
  total_sessions: number
  total_messages: number
  today_questions: number
  today_active_students: number
  daily_trend: { date: string; count: number }[]
}

interface FreqQuestion {
  question: string
  count: number
  last_asked: number | null
}

interface ChatSession {
  session_id: string
  title: string
  student: { id: string; username: string; display_name: string }
  message_count: number
  last_message_at: number | null
  created_at: number
}

interface ChatMessage {
  id: string
  role: string
  content: string
  msg_type: string
  created_at: number
}

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

export default function TeacherPage({ user, onBack }: Props) {
  const [courses, setCourses] = useState<KB[]>([])
  const [selectedKB, setSelectedKB] = useState<KB | null>(null)
  const [students, setStudents] = useState<Student[]>([])
  const [loadingCourses, setLoadingCourses] = useState(false)
  const [loadingStudents, setLoadingStudents] = useState(false)
  const [generatingCode, setGeneratingCode] = useState(false)
  const [indexSubmitting, setIndexSubmitting] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [joinCodeInput, setJoinCodeInput] = useState('')
  const [joiningCourse, setJoiningCourse] = useState(false)

  // Analytics state
  const [overview, setOverview] = useState<OverviewData | null>(null)
  const [freqQuestions, setFreqQuestions] = useState<FreqQuestion[]>([])
  const [expandedQ, setExpandedQ] = useState<number | null>(null)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [chatPage, setChatPage] = useState(1)
  const [chatFilterStudent, setChatFilterStudent] = useState<string>('')
  const [viewingMessages, setViewingMessages] = useState<ChatMessage[] | null>(null)
  const [viewingSessionTitle, setViewingSessionTitle] = useState('')

  const [showCreate, setShowCreate] = useState(false)
  const [newId, setNewId] = useState('')
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newIcon, setNewIcon] = useState('📘')
  const [newIndexBackend, setNewIndexBackend] = useState('lightrag')
  const [creating, setCreating] = useState(false)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── 加载课程列表 ─────────────────────────────────────────────────────────────

  const loadCourses = async () => {
    setLoadingCourses(true)
    try {
      const list = await apiFetch('/teacher/courses') as KB[]
      setCourses(list)
      // 只同步列表里的标量字段，保留 selectedKB 的 files（列表接口不返回 files）
      setSelectedKB(prev => {
        if (!prev) return null
        const fresh = list.find(c => c.course_id === prev.course_id)
        if (!fresh) return null
        return { ...prev, ...fresh, files: prev.files }
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoadingCourses(false)
    }
  }

  // ── 加载知识库详情（含文件列表） ─────────────────────────────────────────────

  const loadKBDetail = async (courseId: string) => {
    try {
      const data = await apiFetch(`/teacher/courses/${courseId}`) as KB
      setSelectedKB(data)
      setCourses(prev => prev.map(c => c.course_id === courseId ? { ...c, ...data } : c))
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载详情失败')
    }
  }

  useEffect(() => { void loadCourses() }, [])

  // ── 轮询正在索引的课程 ────────────────────────────────────────────────────────

  const hasIndexing = courses.some(c => c.status === 'indexing')
  const selectedCourseId = selectedKB?.course_id ?? null

  useEffect(() => {
    if (hasIndexing && !pollRef.current) {
      pollRef.current = setInterval(() => {
        void loadCourses()
        if (selectedCourseId) void loadKBDetail(selectedCourseId)
      }, 2000)
    } else if (!hasIndexing && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [hasIndexing, selectedCourseId])

  // ── 选中课程 ─────────────────────────────────────────────────────────────────

  const selectCourse = (c: KB) => {
    setStudents([])
    setOverview(null)
    setFreqQuestions([])
    setChatSessions([])
    setChatPage(1)
    setChatFilterStudent('')
    setViewingMessages(null)
    setExpandedQ(null)
    void loadKBDetail(c.course_id)
    void loadStudents(c.course_id)
    void loadOverview(c.course_id)
    void loadFreqQuestions(c.course_id)
    void loadChatSessions(c.course_id)
  }

  const loadStudents = async (courseId: string) => {
    setLoadingStudents(true)
    try {
      const list = await apiFetch(`/teacher/courses/${courseId}/students`) as Student[]
      setStudents(list)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载学生失败')
    } finally {
      setLoadingStudents(false)
    }
  }

  // ── Analytics loaders ──────────────────────────────────────────────────────────

  const loadOverview = async (courseId: string) => {
    try {
      const data = await apiFetch(`/teacher/courses/${courseId}/analytics/overview`) as OverviewData
      setOverview(data)
    } catch { setOverview(null) }
  }

  const loadFreqQuestions = async (courseId: string) => {
    try {
      const data = await apiFetch(`/teacher/courses/${courseId}/analytics/frequent-questions`) as { questions: FreqQuestion[] }
      setFreqQuestions(data.questions ?? [])
    } catch { setFreqQuestions([]) }
  }

  const loadChatSessions = async (courseId: string, studentId?: string, page = 1) => {
    try {
      const params = new URLSearchParams({ page: String(page), page_size: '20' })
      if (studentId) params.set('student_id', studentId)
      const data = await apiFetch(`/teacher/courses/${courseId}/analytics/student-chats?${params}`) as { sessions: ChatSession[] }
      setChatSessions(data.sessions ?? [])
    } catch { setChatSessions([]) }
  }

  const loadSessionMessages = async (courseId: string, sessionId: string, title: string) => {
    try {
      const data = await apiFetch(`/teacher/courses/${courseId}/analytics/sessions/${sessionId}/messages`) as { messages: ChatMessage[] }
      setViewingMessages(data.messages ?? [])
      setViewingSessionTitle(title)
    } catch { setViewingMessages(null) }
  }

  // ── KbDetailPanel 回调 ────────────────────────────────────────────────────────

  const handleDeleteKB = async (courseId: string) => {
    try {
      await apiFetch(`/teacher/courses/${courseId}`, { method: 'DELETE' })
      setCourses(prev => prev.filter(c => c.course_id !== courseId))
      if (selectedKB?.course_id === courseId) setSelectedKB(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleDeleteFile = async (courseId: string, fileId: string) => {
    try {
      const res = await apiFetch(`/teacher/courses/${courseId}/files/${fileId}`, { method: 'DELETE' }) as { remaining_files: number }
      // 立即从本地 files 中移除，不等后端全量刷新
      setSelectedKB(prev => {
        if (!prev || prev.course_id !== courseId) return prev
        return { ...prev, file_count: res.remaining_files, files: prev.files?.filter(f => f.id !== fileId) }
      })
      setCourses(prev => prev.map(c => c.course_id === courseId ? { ...c, file_count: res.remaining_files } : c))
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除文件失败')
    }
  }

  const handleIndex = async (courseId: string, force = false, resume = false, backend?: string) => {
    setIndexSubmitting(courseId)
    setError('')
    try {
      const params = new URLSearchParams()
      if (force) params.set('force', 'true')
      if (resume) params.set('resume', 'true')
      if (backend) params.set('backend', backend)
      const qs = params.toString()
      await apiFetch(`/teacher/courses/${courseId}/index${qs ? '?' + qs : ''}`, { method: 'POST' })
      // 乐观更新：接口已提前写 indexing（见后端 index_course），这里同步本地状态，
      // 按钮区瞬间从"开始索引"切到"暂停/终止"，不必等 loadCourses 回来（消除闪烁）。
      setSelectedKB(prev => prev && prev.course_id === courseId
        ? { ...prev, status: 'indexing', progress_msg: '准备中…' } : prev)
      setCourses(prev => prev.map(c => c.course_id === courseId ? { ...c, status: 'indexing' } : c))
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '启动索引失败')
    } finally {
      setIndexSubmitting(null)
    }
  }

  const handlePauseIndex = async (courseId: string, backend?: string) => {
    try {
      const qs = backend ? `?backend=${encodeURIComponent(backend)}` : ''
      await apiFetch(`/teacher/courses/${courseId}/index/pause${qs}`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '暂停失败')
    }
  }

  const handleStopIndex = async (courseId: string, backend?: string) => {
    try {
      const qs = backend ? `?backend=${encodeURIComponent(backend)}` : ''
      await apiFetch(`/teacher/courses/${courseId}/index/stop${qs}`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '终止失败')
    }
  }

  // ── 课程码 ────────────────────────────────────────────────────────────────────

  const refreshJoinCode = async () => {
    if (!selectedKB) return
    setGeneratingCode(true)
    try {
      const res = await apiFetch(`/teacher/courses/${selectedKB.course_id}/join-code`, {
        method: 'POST',
      }) as { join_code: string }
      const updated = { ...selectedKB, join_code: res.join_code }
      setSelectedKB(updated)
      setCourses(prev => prev.map(c => c.course_id === selectedKB.course_id ? updated : c))
    } catch (e) {
      setError(e instanceof Error ? e.message : '生成失败')
    } finally {
      setGeneratingCode(false)
    }
  }

  // ── 学生管理 ──────────────────────────────────────────────────────────────────

  const removeStudent = async (studentId: string) => {
    if (!selectedKB) return
    try {
      await apiFetch(`/teacher/courses/${selectedKB.course_id}/students/${studentId}`, { method: 'DELETE' })
      setStudents(prev => prev.filter(s => s.id !== studentId))
    } catch (e) {
      setError(e instanceof Error ? e.message : '操作失败')
    }
  }

  // ── 新建课程 ──────────────────────────────────────────────────────────────────

  const createCourse = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newId.trim() || !newName.trim()) return
    setCreating(true)
    try {
      await apiFetch('/teacher/courses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          course_id: newId.trim(),
          name: newName.trim(),
          description: newDesc.trim(),
          icon: newIcon.trim() || '📘',
          index_backend: newIndexBackend,
        }),
      })
      setShowCreate(false)
      setNewId(''); setNewName(''); setNewDesc(''); setNewIcon('📘'); setNewIndexBackend('lightrag')
      await loadCourses()
    } catch (e) {
      setError(e instanceof Error ? e.message : '创建失败')
    } finally {
      setCreating(false)
    }
  }

  // ── 学生用课程码加入 ──────────────────────────────────────────────────────────

  const joinByCourseCode = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!joinCodeInput.trim()) return
    setJoiningCourse(true)
    setError('')
    try {
      const res = await apiFetch('/courses/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ join_code: joinCodeInput.trim() }),
      }) as { name: string; already_enrolled: boolean }
      setError(res.already_enrolled ? `已在课程「${res.name}」中` : `成功加入「${res.name}」`)
      setJoinCodeInput('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '加入失败')
    } finally {
      setJoiningCourse(false)
    }
  }

  // ── 渲染 ──────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      {/* Header */}
      <header className="bg-surface border-b border-line px-4 md:px-6 py-3 md:py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-muted hover:text-ink text-sm transition">
            ← 返回
          </button>
          <h1 className="text-lg font-bold text-ink">教师工作台</h1>
          <span className="text-xs bg-surface-2 text-ink px-2 py-0.5 rounded-full">
            {user.display_name || user.username}
          </span>
        </div>
        <button
          onClick={() => { setShowCreate(true); setError('') }}
          className="px-4 py-2 bg-accent text-white text-sm rounded-[var(--radius)] hover:bg-accent-2 transition"
        >
          + 新建课程
        </button>
      </header>

      {error && (
        <div className="mx-4 md:mx-6 mt-4 px-4 py-2 bg-info-bg text-info-fg rounded-[var(--radius)] text-sm flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError('')} className="ml-2 text-info-fg hover:text-info-fg">✕</button>
        </div>
      )}

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden p-4 md:p-6 gap-4 md:gap-6">
        {/* 课程列表 */}
        <aside className="w-full md:w-72 flex-shrink-0 flex flex-col gap-3 max-h-48 md:max-h-none overflow-y-auto">
          <h2 className="text-sm font-semibold text-ink-soft uppercase tracking-wide">我的课程</h2>
          {loadingCourses ? (
            <p className="text-sm text-muted">加载中...</p>
          ) : courses.length === 0 ? (
            <p className="text-sm text-muted">暂无课程，点击右上角新建</p>
          ) : (
            courses.map(c => (
              <button
                key={c.course_id}
                onClick={() => selectCourse(c)}
                className={`w-full text-left px-4 py-3 rounded-[var(--radius)] border transition ${
                  selectedKB?.course_id === c.course_id
                    ? 'border-ink bg-surface-2'
                    : 'border-line bg-surface hover:border-line hover:bg-canvas'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xl">{c.icon}</span>
                  <span className="text-sm font-medium text-ink truncate">{c.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_COLOR[c.status] ?? 'bg-surface-2 text-ink-soft'}`}>
                    {STATUS_LABEL[c.status] ?? c.status}
                  </span>
                  <span className="text-xs text-muted">{c.file_count} 文件</span>
                </div>
              </button>
            ))
          )}
        </aside>

        {/* 课程详情 */}
        <main className="flex-1 overflow-y-auto space-y-6">
          {!selectedKB ? (
            <div className="flex flex-col items-center justify-center h-64 text-muted gap-2">
              <span className="text-4xl"></span>
              <p className="text-sm">点击左侧课程查看详情</p>
            </div>
          ) : (
            <>
              {/* 知识库管理 — 复用共享组件 */}
              <KbDetailPanel
                kb={selectedKB}
                apiBase="/teacher/courses"
                onDelete={handleDeleteKB}
                onDeleteFile={handleDeleteFile}
                onIndex={handleIndex}
                onPause={handlePauseIndex}
                onStop={handleStopIndex}
                indexSubmitting={indexSubmitting === selectedKB.course_id}
                onRefresh={() => loadKBDetail(selectedKB.course_id)}
                onUploaded={async () => { await loadKBDetail(selectedKB.course_id); await loadCourses() }}
                onUpdated={async () => { await loadKBDetail(selectedKB.course_id); await loadCourses() }}
              />

              {/* 学生入课方式：课程码 + 二维码 + 分享链接（置顶展示，避免埋在底部找不到） */}
              <JoinCodeShareSection
                kb={selectedKB}
                onReset={refreshJoinCode}
                resetting={generatingCode}
              />

              {/* ── 活跃度概览 ────────────────────────────────────────────── */}
              {overview && (
                <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                  <h3 className="font-semibold text-ink mb-4 flex items-center gap-2">
                    <span></span> 活跃度概览
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    {[
                      { label: '总学生', value: overview.total_students, color: 'bg-info-bg text-info-fg' },
                      { label: '总会话', value: overview.total_sessions, color: 'bg-ok-bg text-ok-fg' },
                      { label: '今日提问', value: overview.today_questions, color: 'bg-warn-bg text-warn-fg' },
                      { label: '今日活跃', value: overview.today_active_students, color: 'bg-info-bg text-info-fg' },
                    ].map(card => (
                      <div key={card.label} className={`rounded-[var(--radius)] p-4 ${card.color}`}>
                        <p className="text-xs opacity-70 mb-1">{card.label}</p>
                        <p className="text-2xl font-bold">{card.value}</p>
                      </div>
                    ))}
                  </div>
                  {overview.daily_trend.length > 0 && (
                    <div>
                      <p className="text-xs text-ink-soft mb-2">近 7 天提问趋势</p>
                      <div className="flex items-end gap-1 h-24">
                        {(() => {
                          const maxC = Math.max(...overview.daily_trend.map(d => d.count), 1)
                          return overview.daily_trend.map(d => (
                            <div key={d.date} className="flex-1 flex flex-col items-center gap-1">
                              <span className="text-[10px] text-ink-soft">{d.count}</span>
                              <div
                                className="w-full bg-accent rounded-t"
                                style={{ height: `${(d.count / maxC) * 100}%`, minHeight: d.count > 0 ? 4 : 1 }}
                              />
                              <span className="text-[10px] text-muted">{d.date.slice(5)}</span>
                            </div>
                          ))
                        })()}
                      </div>
                    </div>
                  )}
                </section>
              )}

              {/* ── 高频问题 ──────────────────────────────────────────────── */}
              {freqQuestions.length > 0 && (
                <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                  <h3 className="font-semibold text-ink mb-4 flex items-center gap-2">
                    <span></span> 高频问题
                    <span className="text-xs bg-surface-2 text-ink-soft px-2 py-0.5 rounded-full">{freqQuestions.length} 条</span>
                  </h3>
                  <div className="divide-y divide-line">
                    {freqQuestions.map((q, i) => (
                      <div key={i} className="py-2.5">
                        <div className="flex items-center justify-between cursor-pointer" onClick={() => setExpandedQ(expandedQ === i ? null : i)}>
                          <span className="text-sm text-ink truncate flex-1 mr-3">
                            {expandedQ === i ? q.question : q.question.length > 60 ? q.question.slice(0, 60) + '…' : q.question}
                          </span>
                          <div className="flex items-center gap-3 flex-shrink-0">
                            <span className="text-xs font-medium bg-danger-bg text-danger-fg px-2 py-0.5 rounded-full">{q.count} 次</span>
                            {q.last_asked && (
                              <span className="text-[11px] text-muted">{new Date(q.last_asked * 1000).toLocaleDateString()}</span>
                            )}
                          </div>
                        </div>
                        {expandedQ === i && q.question.length > 60 && (
                          <p className="mt-1 text-sm text-ink-soft bg-canvas rounded-[var(--radius)] p-3">{q.question}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* ── 学生问答记录 ──────────────────────────────────────────── */}
              <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                <h3 className="font-semibold text-ink mb-4 flex items-center gap-2">
                  <span></span> 学生问答记录
                </h3>

                {/* 学生筛选 */}
                {students.length > 0 && (
                  <div className="mb-4 flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => { setChatFilterStudent(''); setChatPage(1); void loadChatSessions(selectedKB.course_id) }}
                      className={`px-3 py-1.5 text-xs rounded-[var(--radius)] transition ${!chatFilterStudent ? 'bg-accent text-white' : 'bg-surface-2 text-ink-soft hover:bg-surface-2'}`}
                    >
                      全部
                    </button>
                    {students.map(s => (
                      <button
                        key={s.id}
                        onClick={() => { setChatFilterStudent(s.id); setChatPage(1); void loadChatSessions(selectedKB.course_id, s.id) }}
                        className={`px-3 py-1.5 text-xs rounded-[var(--radius)] transition ${chatFilterStudent === s.id ? 'bg-accent text-white' : 'bg-surface-2 text-ink-soft hover:bg-surface-2'}`}
                      >
                        {s.display_name || s.username}
                      </button>
                    ))}
                  </div>
                )}

                {chatSessions.length === 0 ? (
                  <p className="text-sm text-muted">暂无问答记录</p>
                ) : (
                  <>
                    <div className="divide-y divide-line">
                      {chatSessions.map(cs => (
                        <div key={cs.session_id} className="py-3 flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs bg-surface-2 text-ink-soft px-1.5 py-0.5 rounded">
                                {cs.student.display_name || cs.student.username || '匿名'}
                              </span>
                              <span className="text-sm font-medium text-ink truncate">{cs.title}</span>
                            </div>
                            <div className="flex items-center gap-3 text-[11px] text-muted">
                              <span>{cs.message_count} 条消息</span>
                              {cs.last_message_at && <span>{new Date(cs.last_message_at * 1000).toLocaleString()}</span>}
                            </div>
                          </div>
                          <button
                            onClick={() => void loadSessionMessages(selectedKB.course_id, cs.session_id, cs.title)}
                            className="ml-3 px-3 py-1.5 text-xs bg-surface-2 text-ink rounded-[var(--radius)] hover:bg-surface-2 transition flex-shrink-0"
                          >
                            查看对话
                          </button>
                        </div>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 mt-4">
                      <button
                        disabled={chatPage <= 1}
                        onClick={() => { const p = chatPage - 1; setChatPage(p); void loadChatSessions(selectedKB.course_id, chatFilterStudent || undefined, p) }}
                        className="px-3 py-1 text-xs bg-surface-2 rounded-[var(--radius)] hover:bg-surface-2 disabled:opacity-40 transition"
                      >
                        上一页
                      </button>
                      <span className="text-xs text-ink-soft">第 {chatPage} 页</span>
                      <button
                        disabled={chatSessions.length < 20}
                        onClick={() => { const p = chatPage + 1; setChatPage(p); void loadChatSessions(selectedKB.course_id, chatFilterStudent || undefined, p) }}
                        className="px-3 py-1 text-xs bg-surface-2 rounded-[var(--radius)] hover:bg-surface-2 disabled:opacity-40 transition"
                      >
                        下一页
                      </button>
                    </div>
                  </>
                )}
              </section>

              {/* 学生列表 */}
              <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
                <h3 className="font-semibold text-ink mb-4 flex items-center gap-2">
                  <span></span> 学生列表
                  <span className="text-xs bg-surface-2 text-ink-soft px-2 py-0.5 rounded-full">
                    {students.length} 人
                  </span>
                </h3>
                {loadingStudents ? (
                  <p className="text-sm text-muted">加载中...</p>
                ) : students.length === 0 ? (
                  <p className="text-sm text-muted">暂无学生，分享课程码邀请学生加入</p>
                ) : (
                  <div className="divide-y divide-line">
                    {students.map(s => (
                      <div key={s.id} className="flex items-center justify-between py-2.5">
                        <div>
                          <span className="text-sm font-medium text-ink">
                            {s.display_name || s.username}
                          </span>
                          <span className="text-xs text-muted ml-2">@{s.username}</span>
                        </div>
                        <button
                          onClick={() => removeStudent(s.id)}
                          className="text-xs text-danger-fg hover:text-danger-fg transition"
                        >
                          移除
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </main>
      </div>

      {/* 新建课程 Modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-surface rounded-[var(--radius-lg)] shadow-2xl w-full max-w-md p-8">
            <h2 className="text-lg font-bold text-ink mb-6">新建课程</h2>
            <form onSubmit={createCourse} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-ink-soft mb-1">课程 ID（英文/数字/下划线）</label>
                <input
                  value={newId}
                  onChange={e => setNewId(e.target.value)}
                  className="w-full rounded-[var(--radius)] border border-line px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-ink focus:ring-2 focus:ring-ink/10 transition"
                  placeholder="my_course_2026"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-soft mb-1">课程名称</label>
                <input
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  className="w-full rounded-[var(--radius)] border border-line px-4 py-2.5 text-sm focus:outline-none focus:border-ink focus:ring-2 focus:ring-ink/10 transition"
                  placeholder="线性代数 A"
                  required
                />
              </div>
              <div className="flex gap-3">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-ink-soft mb-1">简介</label>
                  <input
                    value={newDesc}
                    onChange={e => setNewDesc(e.target.value)}
                    className="w-full rounded-[var(--radius)] border border-line px-4 py-2.5 text-sm focus:outline-none focus:border-ink focus:ring-2 focus:ring-ink/10 transition"
                    placeholder="课程描述（可选）"
                  />
                </div>
                <div className="w-24">
                  <label className="block text-sm font-medium text-ink-soft mb-1">图标</label>
                  <input
                    value={newIcon}
                    onChange={e => setNewIcon(e.target.value)}
                    className="w-full rounded-[var(--radius)] border border-line px-4 py-2.5 text-sm text-center focus:outline-none focus:border-ink focus:ring-2 focus:ring-ink/10 transition"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-soft mb-1">索引后端</label>
                <select
                  value={newIndexBackend}
                  onChange={e => setNewIndexBackend(e.target.value)}
                  className="w-full rounded-[var(--radius)] border border-line px-4 py-2.5 text-sm focus:outline-none focus:border-ink focus:ring-2 focus:ring-ink/10 transition"
                >
                  <option value="lightrag">LightRAG（知识图谱，慢但支持多跳）</option>
                  <option value="llamaindex_pg">pgvector（快速向量，分钟级索引）</option>
                </select>
              </div>
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCreate(false)}
                  className="flex-1 py-2.5 rounded-[var(--radius)] border border-line text-ink-soft text-sm hover:bg-canvas transition"
                >
                  取消
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="flex-1 py-2.5 rounded-[var(--radius)] bg-accent text-white text-sm font-medium hover:bg-accent-2 disabled:opacity-50 transition"
                >
                  {creating ? '创建中...' : '创建'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 查看对话 Modal */}
      {viewingMessages && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50" onClick={() => setViewingMessages(null)}>
          <div className="bg-surface rounded-[var(--radius-lg)] shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-line flex items-center justify-between">
              <h2 className="text-base font-bold text-ink truncate">{viewingSessionTitle || '对话详情'}</h2>
              <button onClick={() => setViewingMessages(null)} className="text-muted hover:text-ink-soft text-lg">✕</button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-3">
              {viewingMessages.length === 0 ? (
                <p className="text-sm text-muted text-center">暂无消息</p>
              ) : viewingMessages.map(m => (
                <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-[var(--radius-lg)] px-4 py-3 text-sm whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-accent text-white rounded-br-md'
                      : 'bg-surface-2 text-ink rounded-bl-md'
                  }`}>
                    {m.content}
                    <div className={`text-[10px] mt-1 ${m.role === 'user' ? 'text-ink-soft' : 'text-muted'}`}>
                      {new Date(m.created_at * 1000).toLocaleString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 学生用课程码入课（浮动） */}
      <div className="fixed bottom-6 right-6 bg-surface rounded-[var(--radius-lg)] shadow-xl border border-line p-4 w-72 z-40">
        <p className="text-xs font-semibold text-ink-soft mb-2">用课程码加入课程</p>
        <form onSubmit={joinByCourseCode} className="flex gap-2">
          <input
            value={joinCodeInput}
            onChange={e => setJoinCodeInput(e.target.value.toUpperCase())}
            className="flex-1 rounded-[var(--radius)] border border-line px-3 py-2 text-sm font-mono focus:outline-none focus:border-ink transition"
            placeholder="XXXXXXXX"
            maxLength={16}
          />
          <button
            type="submit"
            disabled={joiningCourse}
            className="px-3 py-2 bg-accent text-white text-sm rounded-[var(--radius)] hover:bg-accent-2 disabled:opacity-50 transition"
          >
            加入
          </button>
        </form>
      </div>
    </div>
  )
}
