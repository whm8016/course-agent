import { useEffect, useRef, useState } from 'react'
import { authHeaders } from '../services/auth'
import type { User } from '../types'

interface Props {
  user: User
  onBack: () => void
}

interface KBFile {
  id: string
  original_name: string
  file_size: number
  status: string
  error_msg: string
  created_at: number
}

interface KB {
  id: string
  course_id: string
  name: string
  description: string
  icon: string
  system_prompt: string
  sort_order: number
  status: 'pending' | 'indexing' | 'ready' | 'error' | 'paused'
  file_count: number
  error_msg: string
  progress: number
  progress_msg: string
  chunks_done: number
  chunks_total: number
  token_estimate: number
  created_at: number
  updated_at: number
  files?: KBFile[]
}

interface SysUser {
  id: string
  username: string
  display_name: string
  is_admin: boolean
  created_at: number
}

const STATUS_LABEL: Record<string, string> = {
  pending: '待索引',
  indexing: '索引中...',
  ready: '就绪',
  error: '错误',
  paused: '已暂停',
}

const STATUS_COLOR: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-700',
  indexing: 'bg-blue-100 text-blue-700',
  ready: 'bg-green-100 text-green-700',
  error: 'bg-red-100 text-red-700',
  paused: 'bg-orange-100 text-orange-700',
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleString('zh-CN')
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

export default function AdminPage({ user, onBack }: Props) {
  const [tab, setTab] = useState<'kb' | 'users'>('kb')
  const [kbs, setKbs] = useState<KB[]>([])
  const [users, setUsers] = useState<SysUser[]>([])
  const [selectedKB, setSelectedKB] = useState<KB | null>(null)
  const [error, setError] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

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

  // ── 轮询正在索引的知识库 ──────────────────────────────────────────────────

  useEffect(() => {
    const indexing = kbs.some(k => k.status === 'indexing')
    if (indexing && !pollRef.current) {
      pollRef.current = setInterval(() => {
        loadKBs()
        if (selectedKB?.status === 'indexing') {
          loadKBDetail(selectedKB.course_id)
        }
      }, 3000)
    } else if (!indexing && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [kbs, selectedKB])

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
    try {
      await apiFetch(`/admin/kb/${courseId}/llamaindex/build`, { method: 'POST' })
      await loadKBs()
      if (selectedKB?.course_id === courseId) await loadKBDetail(courseId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'LlamaIndex 构建索引失败')
    }
  }

  return (
    <div className="flex h-screen bg-slate-50">
      {/* 侧边栏 */}
      <aside className="w-52 bg-white border-r border-slate-200 flex flex-col">
        <div className="px-4 py-5 border-b border-slate-100">
          <h1 className="text-sm font-bold text-slate-800">管理后台</h1>
          <p className="text-xs text-slate-400 mt-0.5">{user.display_name}</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {(['kb', 'users'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition ${
                tab === t
                  ? 'bg-indigo-50 text-indigo-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              {t === 'kb' ? '知识库管理' : '用户管理'}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-slate-100">
          <button
            onClick={onBack}
            className="w-full text-left px-3 py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition"
          >
            返回课程页
          </button>
        </div>
      </aside>

      {/* 主内容 */}
      <main className="flex-1 flex overflow-hidden">
        {tab === 'kb' ? (
          <>
            {/* KB 列表 */}
            <div className="w-80 border-r border-slate-200 bg-white flex flex-col">
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
                      <span className={`text-xs px-1.5 py-0.5 rounded-full ml-1 shrink-0 ${STATUS_COLOR[kb.status]}`}>
                        {STATUS_LABEL[kb.status]}
                      </span>
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
                <KBDetail
                  kb={selectedKB}
                  onDelete={handleDeleteKB}
                  onDeleteFile={handleDeleteFile}
                  onIndex={handleIndex}
                  onPause={handlePauseIndex}
                  onStop={handleStopIndex}
                  onLlamaIndexBuild={handleLlamaIndexBuild}
                  onRefresh={() => loadKBDetail(selectedKB.course_id)}
                  onUploaded={async () => { await loadKBDetail(selectedKB.course_id); await loadKBs() }}
                  onUpdated={async () => { await loadKBDetail(selectedKB.course_id); await loadKBs() }}
                />
              )}
            </div>
          </>
        ) : (
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
                  {users.map(u => (
                    <tr key={u.id} className="hover:bg-slate-50">
                      <td className="px-4 py-3 font-medium text-slate-800">{u.username}</td>
                      <td className="px-4 py-3 text-slate-600">{u.display_name}</td>
                      <td className="px-4 py-3">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${u.is_admin ? 'bg-purple-100 text-purple-700' : 'bg-slate-100 text-slate-500'}`}>
                          {u.is_admin ? '管理员' : '普通用户'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{formatTime(u.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
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

// ── KB 详情子组件 ────────────────────────────────────────────────────────────

function KBDetail({
  kb, onDelete, onDeleteFile, onIndex, onPause, onStop, onLlamaIndexBuild, onRefresh, onUploaded, onUpdated,
}: {
  kb: KB
  onDelete: (courseId: string) => void
  onDeleteFile: (courseId: string, fileId: string) => void
  onIndex: (courseId: string, force?: boolean, resume?: boolean) => void
  onPause: (courseId: string) => void
  onStop: (courseId: string) => void
  onLlamaIndexBuild: (courseId: string) => void
  onRefresh: () => void
  onUploaded: () => void
  onUpdated: () => void
}) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [showEdit, setShowEdit] = useState(false)
  const [editName, setEditName] = useState(kb.name)
  const [editDesc, setEditDesc] = useState(kb.description)
  const [editIcon, setEditIcon] = useState(kb.icon || '📘')
  const [editPrompt, setEditPrompt] = useState(kb.system_prompt || '')
  const [editOrder, setEditOrder] = useState(kb.sort_order ?? 0)
  const [editLoading, setEditLoading] = useState(false)
  const [editError, setEditError] = useState('')

  const handleEditSave = async () => {
    setEditLoading(true)
    setEditError('')
    try {
      await apiFetch(`/admin/kb/${kb.course_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName,
          description: editDesc,
          icon: editIcon,
          system_prompt: editPrompt,
          sort_order: editOrder,
        }),
      })
      setShowEdit(false)
      onUpdated()
    } catch (e) {
      setEditError(e instanceof Error ? e.message : '保存失败')
    } finally {
      setEditLoading(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    setUploading(true)
    setUploadError('')
    try {
      const formData = new FormData()
      Array.from(files).forEach(f => formData.append('files', f))
      const res = await fetch(`/api/admin/kb/${kb.course_id}/upload`, {
        method: 'POST',
        headers: authHeaders(),
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || '上传失败')
      }
      await onUploaded()
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : '上传失败')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-6">
      {/* 头部 */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">
              {kb.icon && <span className="mr-1">{kb.icon}</span>}{kb.name}
            </h2>
            <p className="text-sm text-slate-500 mt-0.5">课程 ID：{kb.course_id}</p>
            {kb.description && <p className="text-sm text-slate-600 mt-1">{kb.description}</p>}
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-sm px-3 py-1 rounded-full ${STATUS_COLOR[kb.status]}`}>
              {STATUS_LABEL[kb.status]}
            </span>
            {/* ready 状态：把"重新索引"折叠成小刷新图标 */}
            {kb.status === 'ready' && (
              <button
                onClick={() => onIndex(kb.course_id, false, false)}
                className="text-slate-400 hover:text-indigo-600 p-1.5 rounded hover:bg-indigo-50 transition"
                title="重新索引（清空已有图谱后从头构建）"
                aria-label="重新索引"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="23 4 23 10 17 10" />
                  <polyline points="1 20 1 14 7 14" />
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
              </button>
            )}
            <button
              onClick={() => { setShowEdit(v => !v); setEditError('') }}
              className="text-xs text-indigo-500 hover:text-indigo-700 px-2 py-1 rounded hover:bg-indigo-50"
            >
              编辑信息
            </button>
            <button
              onClick={onRefresh}
              className="text-xs text-slate-400 hover:text-slate-600 px-2 py-1 rounded hover:bg-slate-100"
              title="刷新当前数据"
            >
              刷新
            </button>
            <button
              onClick={() => onDelete(kb.course_id)}
              className="text-xs text-red-400 hover:text-red-600 px-2 py-1 rounded hover:bg-red-50"
            >
              删除知识库
            </button>
          </div>
        </div>

        {/* 编辑信息折叠面板 */}
        {showEdit && (
          <div className="mt-4 border-t border-slate-100 pt-4 space-y-3">
            {editError && <p className="text-xs text-red-600 bg-red-50 p-2 rounded">{editError}</p>}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">名称</label>
                <input
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">图标（emoji）</label>
                <input
                  value={editIcon}
                  onChange={e => setEditIcon(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400"
                  placeholder="📘"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">描述</label>
              <input
                value={editDesc}
                onChange={e => setEditDesc(e.target.value)}
                className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">
                AI System Prompt
                <span className="ml-1 text-slate-400 font-normal">（AI 助教的角色设定，决定回答风格和范围）</span>
              </label>
              <textarea
                value={editPrompt}
                onChange={e => setEditPrompt(e.target.value)}
                rows={6}
                className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400 resize-y font-mono"
                placeholder="你是一位耐心的课程助教..."
              />
            </div>
            <div className="w-24">
              <label className="block text-xs font-medium text-slate-600 mb-1">排序（小的在前）</label>
              <input
                type="number"
                value={editOrder}
                onChange={e => setEditOrder(Number(e.target.value))}
                className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleEditSave}
                disabled={editLoading}
                className="bg-indigo-600 text-white text-sm px-4 py-1.5 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {editLoading ? '保存中...' : '保存'}
              </button>
              <button
                onClick={() => setShowEdit(false)}
                className="border border-slate-300 text-slate-600 text-sm px-4 py-1.5 rounded-lg hover:bg-slate-50 transition"
              >
                取消
              </button>
            </div>
          </div>
        )}

        {kb.error_msg && (
          <div className="mt-3 p-3 bg-red-50 text-red-600 text-xs rounded-lg">
            错误：{kb.error_msg}
          </div>
        )}

        {/* 索引进度区域 */}
        {kb.status === 'indexing' && (
          <div className="mt-4 space-y-2">
            {/* 进度条 */}
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full transition-all duration-700"
                  style={{ width: `${kb.progress}%` }}
                />
              </div>
              <span className="text-xs font-medium text-indigo-600 w-9 text-right shrink-0">
                {kb.progress}%
              </span>
            </div>
            {/* 当前步骤 */}
            {kb.progress_msg && (
              <p className="text-xs text-slate-500 leading-snug">{kb.progress_msg}</p>
            )}
            {/* 统计行 */}
            <div className="flex flex-wrap gap-4 text-xs text-slate-400">
              {kb.chunks_total > 0 && (
                <span>
                  文本块：<span className="text-slate-600 font-medium">{kb.chunks_done}</span>
                  {' / '}{kb.chunks_total}
                </span>
              )}
              {kb.token_estimate > 0 && (
                <span>
                  估算 Token：<span className="text-slate-600 font-medium">
                    {kb.token_estimate >= 1000
                      ? `${(kb.token_estimate / 1000).toFixed(1)}K`
                      : kb.token_estimate}
                  </span>
                </span>
              )}
            </div>
          </div>
        )}

        {/* 索引完成统计（ready 状态且有 token 数据时显示） */}
        {kb.status === 'ready' && kb.chunks_total > 0 && (
          <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-400">
            <span>共 <span className="text-slate-600 font-medium">{kb.chunks_total}</span> 个文本块</span>
            {kb.token_estimate > 0 && (
              <span>
                累计消耗 Token 约{' '}
                <span className="text-slate-600 font-medium">
                  {kb.token_estimate >= 1000
                    ? `${(kb.token_estimate / 1000).toFixed(1)}K`
                    : kb.token_estimate}
                </span>
              </span>
            )}
          </div>
        )}

        <div className="mt-4 flex items-center flex-wrap gap-2">
          {/* pending：开始索引（大按钮） */}
          {kb.status === 'pending' && (
            <button
              onClick={() => onIndex(kb.course_id, false, false)}
              className="text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
            >
              开始索引
            </button>
          )}

          {/* indexing：LlamaIndex 构建中 —— 只显示等待提示，无暂停/终止 */}
          {kb.status === 'indexing' && kb.progress_msg?.includes('LlamaIndex') && (
            <span className="flex items-center gap-1.5 text-sm text-teal-600">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                className="animate-spin" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
              LlamaIndex 构建中，请稍候…
            </span>
          )}

          {/* indexing：LightRAG 主流程 —— 暂停 + 终止 */}
          {kb.status === 'indexing' && !kb.progress_msg?.includes('LlamaIndex') && (
            <>
              <button
                onClick={() => onPause(kb.course_id)}
                className="flex items-center gap-1 text-sm text-amber-600 border border-amber-300 bg-amber-50 px-3 py-1.5 rounded-lg hover:bg-amber-100 transition"
                title="暂停索引（保留已完成进度，可续传）"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="5" width="4" height="14" rx="1" />
                  <rect x="14" y="5" width="4" height="14" rx="1" />
                </svg>
                暂停
              </button>
              <button
                onClick={() => onStop(kb.course_id)}
                className="flex items-center gap-1 text-sm text-red-600 border border-red-300 bg-red-50 px-3 py-1.5 rounded-lg hover:bg-red-100 transition"
                title="终止索引（清空进度）"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
                终止
              </button>
            </>
          )}

          {/* paused：继续 + 终止 */}
          {kb.status === 'paused' && (
            <>
              <button
                onClick={() => onIndex(kb.course_id, false, true)}
                className="flex items-center gap-1 text-sm bg-indigo-600 text-white px-3 py-1.5 rounded-lg hover:bg-indigo-700 transition"
                title={kb.chunks_done > 0 ? `从第 ${kb.chunks_done} 个文本块继续` : '从头开始'}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="6,4 20,12 6,20" />
                </svg>
                继续{kb.chunks_done > 0 && kb.chunks_total > 0 ? `（${kb.chunks_done}/${kb.chunks_total}）` : ''}
              </button>
              <button
                onClick={() => onStop(kb.course_id)}
                className="flex items-center gap-1 text-sm text-red-600 border border-red-300 bg-red-50 px-3 py-1.5 rounded-lg hover:bg-red-100 transition"
                title="放弃当前进度，恢复待索引状态"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
                终止
              </button>
            </>
          )}

          {/* error：续传 + 重新索引 */}
          {kb.status === 'error' && (
            <>
              {kb.chunks_done > 0 && kb.chunks_total > 0 && (
                <button
                  onClick={() => onIndex(kb.course_id, false, true)}
                  className="text-sm bg-amber-500 text-white px-4 py-2 rounded-lg hover:bg-amber-600 transition"
                  title={`从第 ${kb.chunks_done} 个文本块继续`}
                >
                  续传（{kb.chunks_done}/{kb.chunks_total}）
                </button>
              )}
              <button
                onClick={() => onIndex(kb.course_id, false, false)}
                className="text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
              >
                重新索引
              </button>
            </>
          )}

          {/* LlamaIndex 独立构建按钮：非索引中状态均可触发 */}
          {kb.status !== 'indexing' && kb.file_count > 0 && (
            <button
              onClick={() => onLlamaIndexBuild(kb.course_id)}
              className="flex items-center gap-1.5 text-sm text-teal-700 border border-teal-300 bg-teal-50 px-3 py-1.5 rounded-lg hover:bg-teal-100 transition"
              title="仅构建 LlamaIndex 向量索引（不触发 LightRAG 摄入）"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              LlamaIndex 构建索引
            </button>
          )}

          <p className="text-xs text-slate-400 ml-1">
            {kb.file_count} 个文件 · 更新于 {new Date(kb.updated_at * 1000).toLocaleString('zh-CN')}
          </p>
        </div>
      </div>

      {/* 文件上传区 */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h3 className="font-medium text-slate-700 mb-3">上传文件</h3>
        <p className="text-xs text-slate-400 mb-3">支持 PDF、DOCX、PPTX、TXT、MD，单文件最大 50 MB</p>

        {uploadError && (
          <div className="mb-3 p-2 bg-red-50 text-red-600 text-xs rounded">
            {uploadError}
          </div>
        )}

        <label className="flex items-center justify-center w-full h-24 border-2 border-dashed border-slate-300 rounded-lg cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition">
          <div className="text-center">
            {uploading ? (
              <p className="text-sm text-indigo-600">上传中...</p>
            ) : (
              <>
                <p className="text-sm text-slate-600">点击或拖拽上传文件</p>
                <p className="text-xs text-slate-400 mt-1">支持批量上传</p>
              </>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.docx,.doc,.pptx,.ppt"
            className="hidden"
            onChange={handleUpload}
            disabled={uploading}
          />
        </label>
      </div>

      {/* 文件列表 */}
      {kb.files && kb.files.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-5 py-3 border-b border-slate-100">
            <h3 className="font-medium text-slate-700">文件列表（{kb.files.length}）</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-100">
              <tr>
                {['文件名', '大小', '状态', '上传时间', '操作'].map(h => (
                  <th key={h} className="text-left px-4 py-2 text-xs font-medium text-slate-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {kb.files.map(f => (
                <tr key={f.id} className="hover:bg-slate-50">
                  <td className="px-4 py-2 text-slate-800 max-w-xs truncate" title={f.original_name}>
                    {f.original_name}
                  </td>
                  <td className="px-4 py-2 text-slate-500">{formatBytes(f.file_size)}</td>
                  <td className="px-4 py-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      f.status === 'indexed' ? 'bg-green-100 text-green-700'
                      : f.status === 'error' ? 'bg-red-100 text-red-700'
                      : 'bg-slate-100 text-slate-500'
                    }`}>
                      {f.status === 'indexed' ? '已索引' : f.status === 'error' ? '错误' : '已上传'}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-slate-400">{formatTime(f.created_at)}</td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() => onDeleteFile(kb.course_id, f.id)}
                      className="text-xs text-red-400 hover:text-red-600"
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
