import { useEffect, useRef, useState } from 'react'
import { authHeaders } from '../../services/auth'
import type { User } from '../../types'
import KbDetailPanel, { STATUS_LABEL, STATUS_COLOR } from './KbDetailPanel'
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
  const [copiedCode, setCopiedCode] = useState(false)
  const [llamaIndexSubmitting, setLlamaIndexSubmitting] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [joinCodeInput, setJoinCodeInput] = useState('')
  const [joiningCourse, setJoiningCourse] = useState(false)

  const [showCreate, setShowCreate] = useState(false)
  const [newId, setNewId] = useState('')
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newIcon, setNewIcon] = useState('📘')
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
    void loadKBDetail(c.course_id)
    void loadStudents(c.course_id)
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

  // ── KbDetailPanel 回调 ────────────────────────────────────────────────────────

  const handleDeleteKB = async (courseId: string) => {
    if (!confirm(`确认删除课程「${courseId}」？此操作不可恢复。`)) return
    try {
      await apiFetch(`/teacher/courses/${courseId}`, { method: 'DELETE' })
      setCourses(prev => prev.filter(c => c.course_id !== courseId))
      if (selectedKB?.course_id === courseId) setSelectedKB(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleDeleteFile = async (courseId: string, fileId: string) => {
    if (!confirm('确认删除此文件？')) return
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

  const handleIndex = async (courseId: string, force = false, resume = false) => {
    try {
      const params = new URLSearchParams()
      if (force) params.set('force', 'true')
      if (resume) params.set('resume', 'true')
      const qs = params.toString()
      await apiFetch(`/teacher/courses/${courseId}/index${qs ? '?' + qs : ''}`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '启动索引失败')
    }
  }

  const handlePauseIndex = async (courseId: string) => {
    try {
      await apiFetch(`/teacher/courses/${courseId}/index/pause`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '暂停失败')
    }
  }

  const handleStopIndex = async (courseId: string) => {
    if (!confirm('确认终止索引？已完成的进度将被清除（暂停状态可保留进度）。')) return
    try {
      await apiFetch(`/teacher/courses/${courseId}/index/stop`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '终止失败')
    }
  }

  const handleLlamaIndexBuild = async (courseId: string) => {
    setLlamaIndexSubmitting(courseId)
    setError('')
    try {
      await apiFetch(`/teacher/courses/${courseId}/llamaindex/build`, { method: 'POST' })
      await loadCourses()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'LlamaIndex 构建失败')
    } finally {
      setLlamaIndexSubmitting(null)
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

  const copyCode = (code: string) => {
    void navigator.clipboard.writeText(code)
    setCopiedCode(true)
    setTimeout(() => setCopiedCode(false), 2000)
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
        }),
      })
      setShowCreate(false)
      setNewId(''); setNewName(''); setNewDesc(''); setNewIcon('📘')
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
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 md:px-6 py-3 md:py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-700 text-sm transition">
            ← 返回
          </button>
          <h1 className="text-lg font-bold text-slate-800">教师工作台</h1>
          <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">
            {user.display_name || user.username}
          </span>
        </div>
        <button
          onClick={() => { setShowCreate(true); setError('') }}
          className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition"
        >
          + 新建课程
        </button>
      </header>

      {error && (
        <div className="mx-4 md:mx-6 mt-4 px-4 py-2 bg-blue-50 text-blue-700 rounded-lg text-sm flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError('')} className="ml-2 text-blue-400 hover:text-blue-600">✕</button>
        </div>
      )}

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden p-4 md:p-6 gap-4 md:gap-6">
        {/* 课程列表 */}
        <aside className="w-full md:w-72 flex-shrink-0 flex flex-col gap-3 max-h-48 md:max-h-none overflow-y-auto">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">我的课程</h2>
          {loadingCourses ? (
            <p className="text-sm text-slate-400">加载中...</p>
          ) : courses.length === 0 ? (
            <p className="text-sm text-slate-400">暂无课程，点击右上角新建</p>
          ) : (
            courses.map(c => (
              <button
                key={c.course_id}
                onClick={() => selectCourse(c)}
                className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                  selectedKB?.course_id === c.course_id
                    ? 'border-indigo-400 bg-indigo-50'
                    : 'border-slate-200 bg-white hover:border-indigo-200 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xl">{c.icon}</span>
                  <span className="text-sm font-medium text-slate-800 truncate">{c.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_COLOR[c.status] ?? 'bg-slate-100 text-slate-500'}`}>
                    {STATUS_LABEL[c.status] ?? c.status}
                  </span>
                  <span className="text-xs text-slate-400">{c.file_count} 文件</span>
                </div>
              </button>
            ))
          )}
        </aside>

        {/* 课程详情 */}
        <main className="flex-1 overflow-y-auto space-y-6">
          {!selectedKB ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400 gap-2">
              <span className="text-4xl">👈</span>
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
                onLlamaIndexBuild={handleLlamaIndexBuild}
                llamaIndexSubmitting={llamaIndexSubmitting === selectedKB.course_id}
                onRefresh={() => loadKBDetail(selectedKB.course_id)}
                onUploaded={async () => { await loadKBDetail(selectedKB.course_id); await loadCourses() }}
                onUpdated={async () => { await loadKBDetail(selectedKB.course_id); await loadCourses() }}
              />

              {/* 课程码区 */}
              <section className="bg-white rounded-2xl border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                  <span>🔑</span> 课程码（学生入课凭证）
                </h3>
                {selectedKB.join_code ? (
                  <div className="flex items-center gap-3">
                    <span className="text-3xl font-mono font-bold tracking-widest text-indigo-700 bg-indigo-50 px-6 py-3 rounded-xl border border-indigo-200">
                      {selectedKB.join_code}
                    </span>
                    <button
                      onClick={() => copyCode(selectedKB.join_code!)}
                      className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg transition"
                    >
                      {copiedCode ? '已复制!' : '复制'}
                    </button>
                    <button
                      onClick={refreshJoinCode}
                      disabled={generatingCode}
                      className="px-4 py-2 text-sm bg-orange-100 text-orange-700 hover:bg-orange-200 rounded-lg transition disabled:opacity-50"
                    >
                      {generatingCode ? '生成中...' : '重置'}
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-400">暂无课程码</span>
                    <button
                      onClick={refreshJoinCode}
                      disabled={generatingCode}
                      className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
                    >
                      {generatingCode ? '生成中...' : '生成课程码'}
                    </button>
                  </div>
                )}
                <p className="text-xs text-slate-400 mt-3">
                  学生在注册/登录后，点击侧边栏「用课程码加入」，输入以上代码即可加入本课程。
                </p>
              </section>

              {/* 学生列表 */}
              <section className="bg-white rounded-2xl border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-700 mb-4 flex items-center gap-2">
                  <span>👥</span> 学生列表
                  <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
                    {students.length} 人
                  </span>
                </h3>
                {loadingStudents ? (
                  <p className="text-sm text-slate-400">加载中...</p>
                ) : students.length === 0 ? (
                  <p className="text-sm text-slate-400">暂无学生，分享课程码邀请学生加入</p>
                ) : (
                  <div className="divide-y divide-slate-100">
                    {students.map(s => (
                      <div key={s.id} className="flex items-center justify-between py-2.5">
                        <div>
                          <span className="text-sm font-medium text-slate-700">
                            {s.display_name || s.username}
                          </span>
                          <span className="text-xs text-slate-400 ml-2">@{s.username}</span>
                        </div>
                        <button
                          onClick={() => removeStudent(s.id)}
                          className="text-xs text-red-400 hover:text-red-600 transition"
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
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8">
            <h2 className="text-lg font-bold text-slate-800 mb-6">新建课程</h2>
            <form onSubmit={createCourse} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">课程 ID（英文/数字/下划线）</label>
                <input
                  value={newId}
                  onChange={e => setNewId(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm font-mono focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
                  placeholder="my_course_2026"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">课程名称</label>
                <input
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
                  placeholder="线性代数 A"
                  required
                />
              </div>
              <div className="flex gap-3">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-slate-600 mb-1">简介</label>
                  <input
                    value={newDesc}
                    onChange={e => setNewDesc(e.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
                    placeholder="课程描述（可选）"
                  />
                </div>
                <div className="w-24">
                  <label className="block text-sm font-medium text-slate-600 mb-1">图标</label>
                  <input
                    value={newIcon}
                    onChange={e => setNewIcon(e.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm text-center focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
                  />
                </div>
              </div>
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCreate(false)}
                  className="flex-1 py-2.5 rounded-lg border border-slate-200 text-slate-600 text-sm hover:bg-slate-50 transition"
                >
                  取消
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="flex-1 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition"
                >
                  {creating ? '创建中...' : '创建'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 学生用课程码入课（浮动） */}
      <div className="fixed bottom-6 right-6 bg-white rounded-2xl shadow-xl border border-slate-200 p-4 w-72 z-40">
        <p className="text-xs font-semibold text-slate-500 mb-2">用课程码加入课程</p>
        <form onSubmit={joinByCourseCode} className="flex gap-2">
          <input
            value={joinCodeInput}
            onChange={e => setJoinCodeInput(e.target.value.toUpperCase())}
            className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400 transition"
            placeholder="XXXXXXXX"
            maxLength={16}
          />
          <button
            type="submit"
            disabled={joiningCourse}
            className="px-3 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            加入
          </button>
        </form>
      </div>
    </div>
  )
}
