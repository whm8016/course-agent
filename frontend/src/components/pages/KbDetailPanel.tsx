import { useRef, useState } from 'react'
import { authHeaders } from '../../services/auth'

// ── 共享类型 ─────────────────────────────────────────────────────────────────

export interface KBFile {
  id: string
  original_name: string
  file_size: number
  status: string
  error_msg: string
  created_at: number
}

export interface KB {
  id: string
  course_id: string
  name: string
  description: string
  icon: string
  system_prompt: string
  sort_order: number
  is_visible: boolean
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
  join_code?: string | null
  owner_id?: string
  llamaindex_built?: boolean
  files?: KBFile[]
}

export const STATUS_LABEL: Record<string, string> = {
  pending: '待索引',
  indexing: '索引中...',
  ready: '就绪',
  error: '错误',
  paused: '已暂停',
}

export const STATUS_COLOR: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-700',
  indexing: 'bg-blue-100 text-blue-700',
  ready: 'bg-green-100 text-green-700',
  error: 'bg-red-100 text-red-700',
  paused: 'bg-orange-100 text-orange-700',
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleString('zh-CN')
}

async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers || {}) },
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({})) as { detail?: string }
    throw new Error(data.detail || `请求失败 (${res.status})`)
  }
  return res.json()
}

// ── 确认对话框 ───────────────────────────────────────────────────────────────

interface ConfirmState {
  action: () => void
  title: string
  message: string
  confirmLabel: string
  variant: 'danger' | 'warning'
}

function ConfirmDialog({
  state,
  onClose,
}: {
  state: ConfirmState
  onClose: () => void
}) {
  const isDanger = state.variant === 'danger'
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6" onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-3 mb-4">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${isDanger ? 'bg-red-100' : 'bg-amber-100'}`}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={isDanger ? 'text-red-600' : 'text-amber-600'}>
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-slate-800">{state.title}</h3>
        </div>
        <p className="text-sm text-slate-600 mb-6 leading-relaxed">{state.message}</p>
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg border border-slate-200 text-slate-600 text-sm hover:bg-slate-50 transition"
          >
            取消
          </button>
          <button
            onClick={() => { state.action(); onClose() }}
            className={`flex-1 py-2 rounded-lg text-white text-sm font-medium transition ${isDanger ? 'bg-red-600 hover:bg-red-700' : 'bg-amber-500 hover:bg-amber-600'}`}
          >
            {state.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── 组件 ─────────────────────────────────────────────────────────────────────

interface Props {
  kb: KB
  /**
   * API 基础路径（不含 /api 前缀）。
   * 管理员传 "/admin/kb"，教师传 "/teacher/courses"。
   */
  apiBase: string
  onDelete: (courseId: string) => void
  onDeleteFile: (courseId: string, fileId: string) => void
  onIndex: (courseId: string, force?: boolean, resume?: boolean) => void
  onPause: (courseId: string) => void
  onStop: (courseId: string) => void
  /** 不传则隐藏 LlamaIndex 构建按钮 */
  onLlamaIndexBuild?: (courseId: string) => void
  llamaIndexSubmitting?: boolean
  /** LightRAG 索引（onIndex）提交中 */
  indexSubmitting?: boolean
  onRefresh: () => void
  onUploaded: () => void
  onUpdated: () => void
}

export default function KbDetailPanel({
  kb, apiBase,
  onDelete, onDeleteFile, onIndex, onPause, onStop,
  onLlamaIndexBuild, llamaIndexSubmitting = false, indexSubmitting = false,
  onRefresh, onUploaded, onUpdated,
}: Props) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [showEdit, setShowEdit] = useState(false)
  const [editName, setEditName] = useState(kb.name)
  const [editDesc, setEditDesc] = useState(kb.description)
  const [editIcon, setEditIcon] = useState(kb.icon || '📘')
  const [editPrompt, setEditPrompt] = useState(kb.system_prompt || '')
  const [editOrder, setEditOrder] = useState(kb.sort_order ?? 0)
  const [editVisible, setEditVisible] = useState(kb.is_visible ?? true)
  const [editLoading, setEditLoading] = useState(false)
  const [editError, setEditError] = useState('')

  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null)

  const llamaIndexBuildComplete = !!kb.llamaindex_built
  const hasLightRagIngested = (kb.chunks_total ?? 0) > 0

  const requestConfirm = (state: ConfirmState) => setConfirmState(state)

  const handleEditSave = async () => {
    setEditLoading(true)
    setEditError('')
    try {
      await apiFetch(`${apiBase}/${kb.course_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName,
          description: editDesc,
          icon: editIcon,
          system_prompt: editPrompt,
          sort_order: editOrder,
          is_visible: editVisible,
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
      const res = await fetch(`/api${apiBase}/${kb.course_id}/upload`, {
        method: 'POST',
        headers: authHeaders(),
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({})) as { detail?: string }
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
              onClick={() => requestConfirm({
                action: () => onDelete(kb.course_id),
                title: '删除知识库',
                message: `确认删除课程「${kb.name}」（${kb.course_id}）？此操作不可恢复，所有文件和索引数据将被永久删除。`,
                confirmLabel: '确认删除',
                variant: 'danger',
              })}
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
            <div className="flex items-end gap-4">
              <div className="w-24">
                <label className="block text-xs font-medium text-slate-600 mb-1">排序（小的在前）</label>
                <input
                  type="number"
                  value={editOrder}
                  onChange={e => setEditOrder(Number(e.target.value))}
                  className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-400"
                />
              </div>
              <label className="flex items-center gap-2 cursor-pointer select-none pb-1.5">
                <div
                  onClick={() => setEditVisible(v => !v)}
                  className={`relative w-9 h-5 rounded-full transition-colors ${editVisible ? 'bg-indigo-500' : 'bg-slate-300'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${editVisible ? 'translate-x-4' : ''}`} />
                </div>
                <span className="text-xs font-medium text-slate-600">
                  {editVisible ? '学生可见' : '学生不可见'}
                </span>
              </label>
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
            {kb.progress_msg && (
              <p className="text-xs text-slate-500 leading-snug">{kb.progress_msg}</p>
            )}
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

        {/* 索引完成统计 */}
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

        {/* 两套索引分工说明：讲清谁喂 AI、谁不影响 AI，避免用户建错 */}
        <div className="mt-4 flex items-start gap-2 text-xs text-slate-500 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 leading-relaxed">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0 text-slate-400">
            <circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <span>
            <b className="text-slate-700">本知识库有两套索引：</b>
            ① <b className="text-indigo-600">LightRAG</b>（下方主操作）—— <b>AI 问答 / 出题 / 解题都依赖它</b>，必须摄入后 AI 才能引用课程内容；
            ② <b className="text-teal-600">LlamaIndex</b>（底部「高级」区）—— <b>不影响 AI 对话</b>，仅供管理员手动搜索使用。
          </span>
        </div>

        <div className="mt-4 flex items-center flex-wrap gap-2">
          {/* ready + 已构建：显示完成徽章 */}
          {kb.status === 'ready' && kb.file_count > 0 && hasLightRagIngested && (
            <span className="inline-flex items-center gap-1.5 text-sm font-medium text-indigo-800 bg-indigo-100/80 border border-indigo-200/80 px-3 py-1.5 rounded-lg">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-600">
                <path d="M20 6L9 17l-5-5" />
              </svg>
              LightRAG 已构建
            </span>
          )}

          {/* ready + 未构建：首次摄入按钮 */}
          {kb.status === 'ready' && kb.file_count > 0 && !hasLightRagIngested && (
            <button
              type="button"
              onClick={() => onIndex(kb.course_id, false, false)}
              className="flex items-center gap-1.5 text-sm bg-indigo-600 text-white px-3 py-1.5 rounded-lg hover:bg-indigo-700 transition"
              title="首次摄入 LightRAG 知识图谱"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10" />
                <polyline points="1 20 1 14 7 14" />
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
              </svg>
              启动 LightRAG 摄入
            </button>
          )}

          {/* pending：开始索引 */}
          {kb.status === 'pending' && (
            <button
              onClick={() => onIndex(kb.course_id, false, false)}
              disabled={indexSubmitting}
              className="flex items-center gap-1.5 text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
            >
              {indexSubmitting ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                    className="animate-spin" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                  启动索引中…
                </>
              ) : (
                <>开始索引（LightRAG）</>
              )}
            </button>
          )}

          {/* indexing：LlamaIndex 构建中 — 终止按钮兜底（LlamaIndex 不支持暂停续传，但
              stop 接口对任何 indexing 都直接落库 pending，任务异常卡死时靠它自救） */}
          {kb.status === 'indexing' && kb.progress_msg?.includes('LlamaIndex') && (
            <>
              <span className="flex items-center gap-1.5 text-sm text-teal-600">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                  className="animate-spin" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
                LlamaIndex 构建中，请稍候…
              </span>
              <button
                onClick={() => requestConfirm({
                  action: () => onStop(kb.course_id),
                  title: '终止索引',
                  message: '确认终止索引？已完成的进度将被清除。',
                  confirmLabel: '确认终止',
                  variant: 'danger',
                })}
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

          {/* indexing：LightRAG 主流程 — 暂停 + 终止 */}
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
                onClick={() => requestConfirm({
                  action: () => onStop(kb.course_id),
                  title: '终止索引',
                  message: '确认终止索引？已完成的进度将被清除（暂停状态可保留进度）。',
                  confirmLabel: '确认终止',
                  variant: 'danger',
                })}
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
                onClick={() => requestConfirm({
                  action: () => onStop(kb.course_id),
                  title: '终止索引',
                  message: '确认终止索引？已完成的进度将被清除（暂停状态可保留进度）。',
                  confirmLabel: '确认终止',
                  variant: 'danger',
                })}
                className="flex items-center gap-1 text-sm text-red-600 border border-red-300 bg-red-50 px-3 py-1.5 rounded-lg hover:bg-red-100 transition"
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

          <p className="text-xs text-slate-400 ml-1">
            {kb.file_count} 个文件 · 更新于 {formatTime(kb.updated_at)}
          </p>
        </div>

        {/* 高级 · 向量检索 LlamaIndex：不影响 AI 对话，仅供管理员手动搜索/后续扩展使用。
            故意折叠 + 标红警示，避免用户误以为「建了它 AI 就能答」——AI 只认 LightRAG。 */}
        {onLlamaIndexBuild && kb.file_count > 0 && kb.status !== 'indexing' && (
          <details className="mt-3 border border-teal-100 bg-teal-50/40 rounded-lg px-3 py-2">
            <summary className="text-xs text-teal-700 cursor-pointer hover:text-teal-800 select-none font-medium">
              高级 · LlamaIndex 向量检索{llamaIndexBuildComplete ? '（已构建）' : ''}
            </summary>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="text-xs text-teal-600 leading-relaxed w-full">
                ⚠️ 不影响 AI 对话，仅供管理员手动搜索使用。
              </span>
              {llamaIndexBuildComplete ? (
                <>
                  <span className="inline-flex items-center gap-1.5 text-sm font-medium text-teal-800 bg-teal-100/80 border border-teal-200/80 px-3 py-1.5 rounded-lg">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="text-teal-600">
                      <path d="M20 6L9 17l-5-5" />
                    </svg>
                    LlamaIndex 索引已完成
                  </span>
                  <button
                    type="button"
                    disabled={llamaIndexSubmitting}
                    onClick={() => requestConfirm({
                      action: () => onLlamaIndexBuild(kb.course_id),
                      title: '重新构建 LlamaIndex',
                      message: '重新构建将清空当前 LlamaIndex 索引数据并重新生成。此过程可能需要较长时间，确定继续？',
                      confirmLabel: '确认重建',
                      variant: 'warning',
                    })}
                    className="flex items-center gap-1.5 text-sm text-amber-700 border border-amber-300 bg-amber-50 px-3 py-1.5 rounded-lg hover:bg-amber-100 transition disabled:opacity-50"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="23 4 23 10 17 10" />
                      <polyline points="1 20 1 14 7 14" />
                      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                    </svg>
                    {llamaIndexSubmitting ? '提交重建中…' : '重新构建 LlamaIndex'}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => onLlamaIndexBuild(kb.course_id)}
                  disabled={llamaIndexSubmitting}
                  className="flex items-center gap-1.5 text-sm text-teal-700 border border-teal-300 bg-teal-50 px-3 py-1.5 rounded-lg hover:bg-teal-100 transition disabled:opacity-50"
                >
                  {llamaIndexSubmitting ? (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                        className="animate-spin" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                      </svg>
                      启动构建中…
                    </>
                  ) : (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                      </svg>
                      LlamaIndex 构建索引
                    </>
                  )}
                </button>
              )}
            </div>
          </details>
        )}

        {/* 高级操作（重建 LightRAG）：仅 LightRAG 已摄入时显示。LlamaIndex 重建已并入上方其折叠区。 */}
        {kb.status === 'ready' && kb.file_count > 0 && hasLightRagIngested && (
          <details className="mt-3 border-t border-slate-100 pt-3">
            <summary className="text-xs text-slate-400 cursor-pointer hover:text-slate-600 select-none">
              高级操作（重建 LightRAG）
            </summary>
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => requestConfirm({
                  action: () => onIndex(kb.course_id, false, false),
                  title: '重新构建 LightRAG',
                  message: '重新构建将清空当前知识图谱，按现有文件重新摄入 LightRAG。此过程可能需要较长时间，确定继续？',
                  confirmLabel: '确认重建',
                  variant: 'warning',
                })}
                className="flex items-center gap-1.5 text-sm text-amber-700 border border-amber-300 bg-amber-50 px-3 py-1.5 rounded-lg hover:bg-amber-100 transition"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="23 4 23 10 17 10" />
                  <polyline points="1 20 1 14 7 14" />
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
                重新构建 LightRAG
              </button>
            </div>
          </details>
        )}
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
                      onClick={() => requestConfirm({
                        action: () => onDeleteFile(kb.course_id, f.id),
                        title: '删除文件',
                        message: `确认删除文件「${f.original_name}」？删除后需要重新构建索引。`,
                        confirmLabel: '确认删除',
                        variant: 'danger',
                      })}
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

      {confirmState && (
        <ConfirmDialog state={confirmState} onClose={() => setConfirmState(null)} />
      )}
    </div>
  )
}
