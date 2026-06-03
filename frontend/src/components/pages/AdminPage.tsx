import { useEffect, useRef, useState } from 'react'
import { authHeaders } from '../../services/auth'
import type { User } from '../../types'
import KbDetailPanel, { STATUS_LABEL, STATUS_COLOR, formatTime } from './KbDetailPanel'
import type { KB } from './KbDetailPanel'

interface Props {
  user: User
  onBack: () => void
}

interface SysUser {
  id: string
  username: string
  display_name: string
  role: string
  is_admin: boolean
  created_at: number
}

async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers || {}) },
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `请求失败 (${res.status})`)
  }
  return res.json()
}

interface InviteCode {
  id: string
  code: string
  used_by: string | null
  expires_at: number | null
  created_at: number
}

export default function AdminPage({ user, onBack }: Props) {
  const [tab, setTab] = useState<'kb' | 'users' | 'invites' | 'faq'>('kb')
  const [faqCourseId, setFaqCourseId] = useState('')
  const [faqItems, setFaqItems] = useState<{ question: string; count: number; course_id?: string; course_name?: string }[]>([])
  const [faqThreshold, setFaqThreshold] = useState(3)
  const [faqLoading, setFaqLoading] = useState(false)

  const loadFaq = async (courseId?: string) => {
    setFaqLoading(true)
    try {
      const qs = courseId ? `?course_id=${encodeURIComponent(courseId)}&top_n=50` : '?top_n=50'
      const data = await apiFetch(`/admin/faq${qs}`)
      setFaqItems(data.questions ?? [])
      setFaqThreshold(data.threshold ?? 3)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setFaqLoading(false)
    }
  }
  const [kbs, setKbs] = useState<KB[]>([])
  const [users, setUsers] = useState<SysUser[]>([])
  const [selectedKB, setSelectedKB] = useState<KB | null>(null)
  const [error, setError] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [llamaIndexSubmitting, setLlamaIndexSubmitting] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [inviteCodes, setInviteCodes] = useState<InviteCode[]>([])
  const [inviteLoading, setInviteLoading] = useState(false)
  const [generatingInvite, setGeneratingInvite] = useState(false)
  const [roleChanging, setRoleChanging] = useState<string | null>(null)

  // ── 加载数据 ──────────────────────────────────────────────────────────────

  const loadKBs = async () => {
    try {
      const data = await apiFetch('/admin/kb')
      setKbs(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    }
  }

  const loadUsers = async () => {
    try {
      const data = await apiFetch('/admin/users')
      setUsers(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    }
  }

  const loadInviteCodes = async () => {
    setInviteLoading(true)
    try {
      const data = await apiFetch('/admin/invite-codes') as InviteCode[]
      setInviteCodes(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载邀请码失败')
    } finally {
      setInviteLoading(false)
    }
  }

  const generateInviteCode = async () => {
    setGeneratingInvite(true)
    try {
      await apiFetch('/admin/invite-codes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
      await loadInviteCodes()
    } catch (e) {
      setError(e instanceof Error ? e.message : '生成失败')
    } finally {
      setGeneratingInvite(false)
    }
  }

  const changeUserRole = async (userId: string, newRole: string) => {
    setRoleChanging(userId)
    try {
      await apiFetch(`/admin/users/${userId}/role`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole }),
      })
      await loadUsers()
    } catch (e) {
      setError(e instanceof Error ? e.message : '修改角色失败')
    } finally {
      setRoleChanging(null)
    }
  }

  const loadKBDetail = async (courseId: string) => {
    try {
      const data = await apiFetch(`/admin/kb/${courseId}`)
      setSelectedKB(data)
      setKbs(prev => prev.map(k => k.course_id === courseId ? { ...k, ...data } : k))
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载详情失败')
    }
  }

  useEffect(() => {
    loadKBs()
    loadUsers()
  }, [])

  useEffect(() => {
    if (tab === 'invites') void loadInviteCodes()
  }, [tab])

  // ── 轮询正在索引的知识库（依赖用 hasIndexing + 课程 id，避免每次 loadKBs 后重建定时器）──

  const hasIndexing = kbs.some(k => k.status === 'indexing')
  const selectedCourseId = selectedKB?.course_id ?? null

  useEffect(() => {
    if (hasIndexing && !pollRef.current) {
      pollRef.current = setInterval(() => {
        void loadKBs()
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

  // ── 操作 ──────────────────────────────────────────────────────────────────

  const handleDeleteKB = async (courseId: string) => {
    if (!confirm(`确认删除知识库 "${courseId}"？此操作不可恢复。`)) return
    try {
      await apiFetch(`/admin/kb/${courseId}`, { method: 'DELETE' })
      setKbs(prev => prev.filter(k => k.course_id !== courseId))
      if (selectedKB?.course_id === courseId) setSelectedKB(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleDeleteFile = async (courseId: string, fileId: string) => {
    if (!confirm('确认删除此文件？')) return
    try {
      await apiFetch(`/admin/kb/${courseId}/files/${fileId}`, { method: 'DELETE' })
      await loadKBDetail(courseId)
      await loadKBs()
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
      await apiFetch(`/admin/kb/${courseId}/index${qs ? '?' + qs : ''}`, { method: 'POST' })
      await loadKBs()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '启动索引失败')
    }
  }

  const handlePauseIndex = async (courseId: string) => {
    try {
      await apiFetch(`/admin/kb/${courseId}/index/pause`, { method: 'POST' })
      await loadKBs()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '暂停失败')
    }
  }

  const handleStopIndex = async (courseId: string) => {
    if (!confirm('确认终止索引？已完成的进度将被清除（暂停状态可保留进度）。')) return
    try {
      await apiFetch(`/admin/kb/${courseId}/index/stop`, { method: 'POST' })
      await loadKBs()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : '终止失败')
    }
  }

  const handleLlamaIndexBuild = async (courseId: string) => {
    setLlamaIndexSubmitting(courseId)
    setError('')
    try {
      await apiFetch(`/admin/kb/${courseId}/llamaindex/build`, { method: 'POST' })
      await loadKBs()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'LlamaIndex 构建索引失败')
    } finally {
      setLlamaIndexSubmitting(null)
    }
  }

  return (
    <div className="flex flex-col md:flex-row h-screen bg-slate-50">
      {/* 侧边栏 */}
      <aside className="w-full md:w-52 bg-white border-b md:border-b-0 md:border-r border-slate-200 flex flex-col md:h-full">
        <div className="px-4 py-5 border-b border-slate-100">
          <h1 className="text-sm font-bold text-slate-800">管理后台</h1>
          <p className="text-xs text-slate-400 mt-0.5">{user.display_name}</p>
        </div>
        <nav className="flex md:flex-col md:flex-1 p-2 md:p-3 gap-1 md:space-y-1 overflow-x-auto">
          {(['kb', 'users', 'invites', 'faq'] as const).map(t => (
            <button
              key={t}
              onClick={() => {
                setTab(t)
                if (t === 'faq') loadFaq(faqCourseId || undefined)
              }}
              className={`whitespace-nowrap md:w-full text-left px-3 py-2 rounded-lg text-sm transition ${
                tab === t
                  ? 'bg-indigo-50 text-indigo-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              {t === 'kb' ? '知识库管理' : t === 'users' ? '用户管理' : t === 'invites' ? '教师邀请码' : '高频问题'}
            </button>
          ))}
          <button
            onClick={onBack}
            className="whitespace-nowrap md:hidden text-left px-3 py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition"
          >
            返回
          </button>
        </nav>
        <div className="hidden md:block p-3 border-t border-slate-100">
          <button
            onClick={onBack}
            className="w-full text-left px-3 py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition"
          >
            返回课程页
          </button>
        </div>
      </aside>

      {/* 主内容 */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {tab === 'kb' ? (
          <>
            {/* KB 列表 */}
            <div className="w-full md:w-80 border-b md:border-b-0 md:border-r border-slate-200 bg-white flex flex-col max-h-48 md:max-h-none">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <span className="font-medium text-sm text-slate-700">知识库列表</span>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="text-xs bg-indigo-600 text-white px-3 py-1 rounded-lg hover:bg-indigo-700 transition"
                >
                  + 新建
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {kbs.length === 0 && (
                  <p className="text-xs text-slate-400 text-center mt-8">暂无知识库</p>
                )}
                {kbs.map(kb => (
                  <button
                    key={kb.id}
                    onClick={() => { setSelectedKB(kb); loadKBDetail(kb.course_id) }}
                    className={`w-full text-left px-3 py-3 rounded-lg border transition ${
                      selectedKB?.id === kb.id
                        ? 'border-indigo-200 bg-indigo-50'
                        : 'border-transparent hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-slate-800 truncate">{kb.name}</span>
                      <div className="flex items-center gap-1 ml-1 shrink-0">
                        {!kb.is_visible && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-400" title="已隐藏，学生不可见">
                            隐藏
                          </span>
                        )}
                        <span className={`text-xs px-1.5 py-0.5 rounded-full ${STATUS_COLOR[kb.status]}`}>
                          {STATUS_LABEL[kb.status]}
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-slate-400">{kb.course_id} · {kb.file_count} 个文件</div>
                  </button>
                ))}
              </div>
            </div>

            {/* KB 详情 */}
            <div className="flex-1 overflow-y-auto p-6">
              {error && (
                <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex justify-between">
                  <span>{error}</span>
                  <button onClick={() => setError('')} className="ml-2 text-red-400 hover:text-red-600">✕</button>
                </div>
              )}
              {!selectedKB ? (
                <div className="flex items-center justify-center h-full text-slate-400 text-sm">
                  选择左侧知识库查看详情
                </div>
              ) : (
                <KbDetailPanel
                  kb={selectedKB}
                  apiBase="/admin/kb"
                  onDelete={handleDeleteKB}
                  onDeleteFile={handleDeleteFile}
                  onIndex={handleIndex}
                  onPause={handlePauseIndex}
                  onStop={handleStopIndex}
                  onLlamaIndexBuild={handleLlamaIndexBuild}
                  llamaIndexSubmitting={llamaIndexSubmitting === selectedKB.course_id}
                  onRefresh={() => loadKBDetail(selectedKB.course_id)}
                  onUploaded={async () => { await loadKBDetail(selectedKB.course_id); await loadKBs() }}
                  onUpdated={async () => { await loadKBDetail(selectedKB.course_id); await loadKBs() }}
                />
              )}
            </div>
          </>
        ) : tab === 'faq' ? (
          <div className="flex-1 overflow-y-auto p-6">
            {error && (
              <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex justify-between">
                <span>{error}</span>
                <button onClick={() => setError('')} className="ml-2">✕</button>
              </div>
            )}
            <div className="flex items-center gap-3 mb-4">
              <h2 className="text-lg font-semibold text-slate-800">高频问题</h2>
              <span className="text-xs text-slate-400">（达到 {faqThreshold} 次后缓存答案）</span>
              <div className="flex-1" />
              <input
                type="text"
                placeholder="课程 ID（留空=全部）"
                value={faqCourseId}
                onChange={e => setFaqCourseId(e.target.value)}
                className="border border-slate-200 rounded-lg px-3 py-1.5 text-sm w-48 focus:outline-none focus:ring-2 focus:ring-indigo-300"
              />
              <button
                onClick={() => loadFaq(faqCourseId || undefined)}
                disabled={faqLoading}
                className="text-sm bg-indigo-600 text-white px-4 py-1.5 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {faqLoading ? '加载中...' : '查询'}
              </button>
            </div>
            {faqItems.length === 0 ? (
              <p className="text-sm text-slate-400 text-center mt-16">暂无数据（学生提问后自动统计）</p>
            ) : (
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 border-b border-slate-200">
                    <tr>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 w-12">#</th>
                      {!faqCourseId && (
                        <th className="text-left px-4 py-3 text-xs font-medium text-slate-500">课程</th>
                      )}
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-500">问题</th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 w-24">提问次数</th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 w-20">已缓存</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {faqItems.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="px-4 py-3 text-slate-400">{idx + 1}</td>
                        {!faqCourseId && (
                          <td className="px-4 py-3 text-slate-500 text-xs">
                            <span className="bg-slate-100 px-1.5 py-0.5 rounded">{item.course_name || item.course_id}</span>
                          </td>
                        )}
                        <td className="px-4 py-3 text-slate-800">{item.question}</td>
                        <td className="px-4 py-3">
                          <span className={`font-semibold ${item.count >= faqThreshold ? 'text-indigo-600' : 'text-slate-500'}`}>
                            {item.count}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {item.count >= faqThreshold ? (
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">已缓存</span>
                          ) : (
                            <span className="text-xs text-slate-300">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : tab === 'users' ? (
          <div className="flex-1 overflow-y-auto p-6">
            {error && (
              <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex justify-between">
                <span>{error}</span>
                <button onClick={() => setError('')} className="ml-2">✕</button>
              </div>
            )}
            <h2 className="text-lg font-semibold text-slate-800 mb-4">用户列表</h2>
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    {['用户名', '显示名', '角色', '注册时间'].map(h => (
                      <th key={h} className="text-left px-4 py-3 text-xs font-medium text-slate-500">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {users.map(u => {
                    const displayRole = u.role || (u.is_admin ? 'admin' : 'student')
                    const roleColor: Record<string, string> = {
                      admin: 'bg-purple-100 text-purple-700',
                      teacher: 'bg-green-100 text-green-700',
                      student: 'bg-slate-100 text-slate-500',
                    }
                    const roleLabel: Record<string, string> = {
                      admin: '管理员', teacher: '教师', student: '学生',
                    }
                    return (
                      <tr key={u.id} className="hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium text-slate-800">{u.username}</td>
                        <td className="px-4 py-3 text-slate-600">{u.display_name}</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <span className={`text-xs px-2 py-0.5 rounded-full ${roleColor[displayRole] ?? 'bg-slate-100 text-slate-500'}`}>
                              {roleLabel[displayRole] ?? displayRole}
                            </span>
                            <select
                              value={displayRole}
                              disabled={roleChanging === u.id}
                              onChange={e => void changeUserRole(u.id, e.target.value)}
                              className="text-xs border border-slate-200 rounded px-1.5 py-0.5 bg-white focus:outline-none focus:border-indigo-400"
                            >
                              <option value="student">学生</option>
                              <option value="teacher">教师</option>
                              <option value="admin">管理员</option>
                            </select>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-slate-400">{formatTime(u.created_at)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : tab === 'invites' ? (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-slate-800">教师邀请码</h2>
              <button
                onClick={() => void generateInviteCode()}
                disabled={generatingInvite}
                className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {generatingInvite ? '生成中...' : '+ 生成邀请码'}
              </button>
            </div>
            <p className="text-sm text-slate-500 mb-4">
              将邀请码发给待注册的教师，注册时填写邀请码即可获得教师权限（每码限用一次）。
            </p>
            {inviteLoading ? (
              <p className="text-sm text-slate-400">加载中...</p>
            ) : inviteCodes.length === 0 ? (
              <p className="text-sm text-slate-400">暂无邀请码，点击右上角生成。</p>
            ) : (
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 border-b border-slate-200">
                    <tr>
                      {['邀请码', '状态', '创建时间', '有效期'].map(h => (
                        <th key={h} className="text-left px-4 py-3 text-xs font-medium text-slate-500">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {inviteCodes.map(ic => (
                      <tr key={ic.id} className="hover:bg-slate-50">
                        <td className="px-4 py-3">
                          <span className="font-mono font-bold text-indigo-700 tracking-widest">{ic.code}</span>
                          <button
                            onClick={() => void navigator.clipboard.writeText(ic.code)}
                            className="ml-2 text-xs text-slate-400 hover:text-slate-600"
                          >
                            复制
                          </button>
                        </td>
                        <td className="px-4 py-3">
                          {ic.used_by ? (
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">已使用</span>
                          ) : (
                            <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full">未使用</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-slate-400 text-xs">{formatTime(ic.created_at)}</td>
                        <td className="px-4 py-3 text-slate-400 text-xs">
                          {ic.expires_at ? formatTime(ic.expires_at) : '永不过期'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : null}
      </main>

      {/* 新建知识库 Modal */}
      {showCreateModal && (
        <CreateKBModal
          onClose={() => setShowCreateModal(false)}
          onCreated={async () => { setShowCreateModal(false); await loadKBs() }}
        />
      )}
    </div>
  )
}

// ── 新建知识库 Modal ──────────────────────────────────────────────────────────

function CreateKBModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [courseId, setCourseId] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [icon, setIcon] = useState('📘')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [isVisible, setIsVisible] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      await apiFetch('/admin/kb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          course_id: courseId,
          name,
          description,
          icon,
          system_prompt: systemPrompt,
          is_visible: isVisible,
        }),
      })
      onCreated()
    } catch (e) {
      setError(e instanceof Error ? e.message : '创建失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg p-6 max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">新建知识库</h3>
        {error && <p className="mb-3 text-sm text-red-600 bg-red-50 p-2 rounded">{error}</p>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                课程 ID <span className="text-slate-400 font-normal text-xs">（字母/数字/-/_）</span>
              </label>
              <input
                type="text"
                value={courseId}
                onChange={e => setCourseId(e.target.value)}
                placeholder="例如: circuit_analysis"
                pattern="^[a-zA-Z0-9_\-]+$"
                required
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">图标（emoji）</label>
              <input
                type="text"
                value={icon}
                onChange={e => setIcon(e.target.value)}
                placeholder="📘"
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">知识库名称</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="例如: 电路分析基础"
              required
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">描述（可选）</label>
            <input
              type="text"
              value={description}
              onChange={e => setDescription(e.target.value)}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              AI System Prompt
              <span className="ml-1 text-slate-400 font-normal text-xs">（AI 助教的角色设定，可创建后再填）</span>
            </label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              rows={5}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 resize-y font-mono"
              placeholder="你是一位耐心的课程助教，擅长讲解..."
            />
          </div>
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <div
              onClick={() => setIsVisible(v => !v)}
              className={`relative w-9 h-5 rounded-full transition-colors ${isVisible ? 'bg-indigo-500' : 'bg-slate-300'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${isVisible ? 'translate-x-4' : ''}`} />
            </div>
            <span className="text-sm text-slate-700">
              {isVisible ? '对学生可见' : '对学生隐藏'}
            </span>
          </label>
          <div className="flex gap-3 pt-2">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-indigo-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition"
            >
              {loading ? '创建中...' : '创建'}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex-1 border border-slate-300 text-slate-600 py-2 rounded-lg text-sm hover:bg-slate-50 transition"
            >
              取消
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
