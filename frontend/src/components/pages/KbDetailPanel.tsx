import { useRef, useState } from 'react'
import { authHeaders } from '../../services/auth'
import { STATUS_LABEL, STATUS_COLOR, formatBytes, formatTime } from './kbUtils'

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
  index_backend?: string
  builds?: KbBuild[]
  files?: KBFile[]
}

export interface KbBuild {
  backend: string
  label?: string
  status: 'pending' | 'indexing' | 'ready' | 'error' | 'paused'
  progress: number
  progress_msg: string
  chunks_done: number
  chunks_total: number
  token_estimate: number
  error_msg: string
  updated_at: number
}

// STATUS_LABEL / STATUS_COLOR / formatBytes / formatTime 已移至 ./kbUtils（避免 react-refresh 冲突）

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
      <div className="bg-surface rounded-[var(--radius-lg)] shadow-2xl w-full max-w-sm p-6" onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-3 mb-4">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${isDanger ? 'bg-danger-bg' : 'bg-warn-bg'}`}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={isDanger ? 'text-danger-fg' : 'text-warn-fg'}>
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-ink">{state.title}</h3>
        </div>
        <p className="text-sm text-ink-soft mb-6 leading-relaxed">{state.message}</p>
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-[var(--radius)] border border-line text-ink-soft text-sm hover:bg-canvas transition"
          >
            取消
          </button>
          <button
            onClick={() => { state.action(); onClose() }}
            className={`flex-1 py-2 rounded-[var(--radius)] text-white text-sm font-medium transition ${isDanger ? 'bg-danger-fg hover:bg-danger-fg' : 'bg-warn-fg hover:bg-warn-fg'}`}
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
  onIndex: (courseId: string, force?: boolean, resume?: boolean, backend?: string) => void
  onPause: (courseId: string, backend?: string) => void
  onStop: (courseId: string, backend?: string) => void
  /** LightRAG 索引（onIndex）提交中 */
  indexSubmitting?: boolean
  onRefresh: () => void
  onUploaded: () => void
  onUpdated: () => void
}

export default function KbDetailPanel({
  kb, apiBase,
  onDelete, onDeleteFile, onIndex, onPause, onStop,
  indexSubmitting = false,
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
      <div className="bg-surface rounded-[var(--radius)] border border-line p-5">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">
              {kb.icon && <span className="mr-1">{kb.icon}</span>}{kb.name}
            </h2>
            <p className="text-sm text-ink-soft mt-0.5">课程 ID：{kb.course_id}</p>
            {kb.description && <p className="text-sm text-ink-soft mt-1">{kb.description}</p>}
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-sm px-3 py-1 rounded-full ${STATUS_COLOR[kb.status]}`}>
              {STATUS_LABEL[kb.status]}
            </span>
            <button
              onClick={() => { setShowEdit(v => !v); setEditError('') }}
              className="text-xs text-ink hover:text-ink px-2 py-1 rounded hover:bg-surface-2"
            >
              编辑信息
            </button>
            <button
              onClick={onRefresh}
              className="text-xs text-muted hover:text-ink-soft px-2 py-1 rounded hover:bg-surface-2"
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
              className="text-xs text-danger-fg hover:text-danger-fg px-2 py-1 rounded hover:bg-danger-bg"
            >
              删除知识库
            </button>
          </div>
        </div>

        {/* 编辑信息折叠面板 */}
        {showEdit && (
          <div className="mt-4 border-t border-line pt-4 space-y-3">
            {editError && <p className="text-xs text-danger-fg bg-danger-bg p-2 rounded">{editError}</p>}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-ink-soft mb-1">名称</label>
                <input
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  className="w-full border border-ink-soft rounded-[var(--radius)] px-2 py-1.5 text-sm focus:outline-none focus:border-ink"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-ink-soft mb-1">图标（emoji）</label>
                <input
                  value={editIcon}
                  onChange={e => setEditIcon(e.target.value)}
                  className="w-full border border-ink-soft rounded-[var(--radius)] px-2 py-1.5 text-sm focus:outline-none focus:border-ink"
                  placeholder="📘"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-ink-soft mb-1">描述</label>
              <input
                value={editDesc}
                onChange={e => setEditDesc(e.target.value)}
                className="w-full border border-ink-soft rounded-[var(--radius)] px-2 py-1.5 text-sm focus:outline-none focus:border-ink"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-ink-soft mb-1">
                AI System Prompt
                <span className="ml-1 text-muted font-normal">（AI 助教的角色设定，决定回答风格和范围）</span>
              </label>
              <textarea
                value={editPrompt}
                onChange={e => setEditPrompt(e.target.value)}
                rows={6}
                className="w-full border border-ink-soft rounded-[var(--radius)] px-2 py-1.5 text-sm focus:outline-none focus:border-ink resize-y font-mono"
                placeholder="你是一位耐心的课程助教..."
              />
            </div>
            <div className="flex items-end gap-4">
              <div className="w-24">
                <label className="block text-xs font-medium text-ink-soft mb-1">排序（小的在前）</label>
                <input
                  type="number"
                  value={editOrder}
                  onChange={e => setEditOrder(Number(e.target.value))}
                  className="w-full border border-ink-soft rounded-[var(--radius)] px-2 py-1.5 text-sm focus:outline-none focus:border-ink"
                />
              </div>
              <label className="flex items-center gap-2 cursor-pointer select-none pb-1.5">
                <div
                  onClick={() => setEditVisible(v => !v)}
                  className={`relative w-9 h-5 rounded-full transition-colors ${editVisible ? 'bg-accent' : 'bg-ink-soft'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-surface rounded-full shadow transition-transform ${editVisible ? 'translate-x-4' : ''}`} />
                </div>
                <span className="text-xs font-medium text-ink-soft">
                  {editVisible ? '学生可见' : '学生不可见'}
                </span>
              </label>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleEditSave}
                disabled={editLoading}
                className="bg-accent text-white text-sm px-4 py-1.5 rounded-[var(--radius)] hover:bg-accent-2 disabled:opacity-50 transition"
              >
                {editLoading ? '保存中...' : '保存'}
              </button>
              <button
                onClick={() => setShowEdit(false)}
                className="border border-ink-soft text-ink-soft text-sm px-4 py-1.5 rounded-[var(--radius)] hover:bg-canvas transition"
              >
                取消
              </button>
            </div>
          </div>
        )}

        {kb.error_msg && (
          <div className="mt-3 p-3 bg-danger-bg text-danger-fg text-xs rounded-[var(--radius)]">
            错误：{kb.error_msg}
          </div>
        )}

        {/* 索引进度区域 */}
        {kb.status === 'indexing' && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-surface-2 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent rounded-full transition-all duration-700"
                  style={{ width: `${kb.progress}%` }}
                />
              </div>
              <span className="text-xs font-medium text-ink w-9 text-right shrink-0">
                {kb.progress}%
              </span>
            </div>
            {kb.progress_msg && (
              <p className="text-xs text-ink-soft leading-snug">{kb.progress_msg}</p>
            )}
            <div className="flex flex-wrap gap-4 text-xs text-muted">
              {kb.chunks_total > 0 && (
                <span>
                  文本块：<span className="text-ink-soft font-medium">{kb.chunks_done}</span>
                  {' / '}{kb.chunks_total}
                </span>
              )}
              {kb.token_estimate > 0 && (
                <span>
                  估算 Token：<span className="text-ink-soft font-medium">
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
          <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted">
            <span>共 <span className="text-ink-soft font-medium">{kb.chunks_total}</span> 个文本块</span>
            {kb.token_estimate > 0 && (
              <span>
                累计消耗 Token 约{' '}
                <span className="text-ink-soft font-medium">
                  {kb.token_estimate >= 1000
                    ? `${(kb.token_estimate / 1000).toFixed(1)}K`
                    : kb.token_estimate}
                </span>
              </span>
            )}
          </div>
        )}

        {/* 索引说明：LightRAG 是 AI 问答的唯一依赖 */}
        <div className="mt-4 flex items-start gap-2 text-xs text-ink-soft bg-canvas border border-line rounded-[var(--radius)] px-3 py-2 leading-relaxed">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0 text-muted">
            <circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <span>
            <b className="text-ink">知识库索引：</b>
            <b>AI 问答 / 出题 / 解题都依赖它</b>，必须摄入后 AI 才能引用课程内容。一门课可同时构建 LightRAG（知识图谱，多跳）与 pgvector（向量）两套，问答「自动」模式按问题类型选用。
          </span>
        </div>

      </div>

      {/* 双索引构建：LightRAG + pgvector 始终并排，各自独立状态与完整控件（构建/暂停/终止/续传/重建）。
          一门课可同时建两套；没建过的后端显示「未构建 + 构建按钮」。问答 auto 模式按问题类型
          自动选用：多跳→lightrag 图谱，普通→pgvector 向量。 */}
      <div className="mt-4 border-t border-line pt-3">
        <div className="text-xs text-muted mb-2 flex items-center gap-2 flex-wrap">
          <span>双索引（可分别构建，问答「自动」模式按问题类型选用）</span>
          <span className="text-muted/70">{kb.file_count} 个文件 · 更新于 {formatTime(kb.updated_at)}</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {([
            { backend: 'lightrag', label: 'LightRAG' },
            { backend: 'llamaindex_pg', label: 'pgvector' },
          ] as const).map(({ backend, label }) => {
            const b = kb.builds?.find(x => x.backend === backend)
            const notBuilt = !b
            const status = b?.status ?? 'pending'
            const isIndexing = status === 'indexing'
            const done = b?.chunks_done ?? 0
            const total = b?.chunks_total ?? 0
            const stopAction = () => requestConfirm({
              action: () => onStop(kb.course_id, backend),
              title: `终止 ${label} 索引`,
              message: '确认终止索引？已完成的进度将被清除（暂停状态可保留进度）。',
              confirmLabel: '确认终止', variant: 'danger',
            })
            const rebuildAction = () => requestConfirm({
              action: () => onIndex(kb.course_id, false, false, backend),
              title: `重新构建 ${label}`,
              message: `重新构建将清空当前 ${label} 索引并按现有文件重建，可能耗时较长，确定继续？`,
              confirmLabel: '确认重建', variant: 'warning',
            })
            return (
              <div key={backend} className="border border-line rounded-[var(--radius)] p-2.5 bg-surface-2/40">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-medium text-ink">{label}</span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${notBuilt ? 'text-muted bg-canvas' : (STATUS_COLOR[status as keyof typeof STATUS_COLOR] || 'text-muted')}`}>
                    {notBuilt ? '未构建' : ((STATUS_LABEL as Record<string, string>)[status] || status)}
                  </span>
                </div>
                {isIndexing && b && (
                  <div className="mb-1.5">
                    <div className="h-1.5 bg-line/60 rounded-full overflow-hidden">
                      <div className="h-full bg-accent transition-all" style={{ width: `${b.progress || 0}%` }} />
                    </div>
                    <div className="text-xs text-muted mt-1 truncate">{b.progress_msg || '索引中…'}</div>
                  </div>
                )}
                {status === 'error' && b?.error_msg && (
                  <div className="text-xs text-danger-fg mb-1.5 truncate" title={b.error_msg}>{b.error_msg}</div>
                )}
                <div className="flex flex-wrap gap-1.5">
                  {(notBuilt || status === 'pending') && (
                    <button type="button" onClick={() => onIndex(kb.course_id, false, false, backend)}
                      disabled={indexSubmitting}
                      className="text-xs bg-accent text-white px-2.5 py-1 rounded-[var(--radius)] hover:bg-accent-2 transition disabled:opacity-50">
                      构建{label}
                    </button>
                  )}
                  {isIndexing && (
                    <>
                      <button type="button" onClick={() => onPause(kb.course_id, backend)}
                        className="text-xs text-warn-fg border border-warn-fg bg-warn-bg px-2.5 py-1 rounded-[var(--radius)] hover:bg-warn-bg transition">
                        暂停
                      </button>
                      <button type="button" onClick={stopAction}
                        className="text-xs text-danger-fg border border-danger-fg bg-danger-bg px-2.5 py-1 rounded-[var(--radius)] hover:bg-danger-bg transition">
                        终止
                      </button>
                    </>
                  )}
                  {status === 'paused' && (
                    <>
                      <button type="button" onClick={() => onIndex(kb.course_id, false, true, backend)}
                        className="text-xs bg-accent text-white px-2.5 py-1 rounded-[var(--radius)] hover:bg-accent-2 transition">
                        继续{done > 0 && total > 0 ? `（${done}/${total}）` : ''}
                      </button>
                      <button type="button" onClick={stopAction}
                        className="text-xs text-danger-fg border border-danger-fg bg-danger-bg px-2.5 py-1 rounded-[var(--radius)] hover:bg-danger-bg transition">
                        终止
                      </button>
                    </>
                  )}
                  {status === 'error' && (
                    <>
                      {done > 0 && total > 0 && (
                        <button type="button" onClick={() => onIndex(kb.course_id, false, true, backend)}
                          className="text-xs bg-warn-fg text-white px-2.5 py-1 rounded-[var(--radius)] hover:bg-warn-fg transition">
                          续传（{done}/{total}）
                        </button>
                      )}
                      <button type="button" onClick={() => onIndex(kb.course_id, false, false, backend)}
                        className="text-xs bg-accent text-white px-2.5 py-1 rounded-[var(--radius)] hover:bg-accent-2 transition">
                        重新索引
                      </button>
                    </>
                  )}
                  {status === 'ready' && (
                    <button type="button" onClick={rebuildAction}
                      className="text-xs text-warn-fg border border-warn-fg bg-warn-bg px-2.5 py-1 rounded-[var(--radius)] hover:bg-warn-bg transition">
                      重建{label}
                    </button>
                  )}
                </div>
                {status === 'ready' && total > 0 && (
                  <div className="text-[11px] text-muted mt-1.5">{total} 个文本块</div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* 文件上传区 */}
      <div className="bg-surface rounded-[var(--radius)] border border-line p-5">
        <h3 className="font-medium text-ink mb-3">上传文件</h3>
        <p className="text-xs text-muted mb-3">支持 PDF、DOCX、PPTX、TXT、MD，单文件最大 50 MB</p>

        {uploadError && (
          <div className="mb-3 p-2 bg-danger-bg text-danger-fg text-xs rounded">
            {uploadError}
          </div>
        )}

        <label className="flex items-center justify-center w-full h-24 border-2 border-dashed border-ink-soft rounded-[var(--radius)] cursor-pointer hover:border-ink hover:bg-surface-2 transition">
          <div className="text-center">
            {uploading ? (
              <p className="text-sm text-ink">上传中...</p>
            ) : (
              <>
                <p className="text-sm text-ink-soft">点击或拖拽上传文件</p>
                <p className="text-xs text-muted mt-1">支持批量上传</p>
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
        <div className="bg-surface rounded-[var(--radius)] border border-line overflow-hidden">
          <div className="px-5 py-3 border-b border-line">
            <h3 className="font-medium text-ink">文件列表（{kb.files.length}）</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-canvas border-b border-line">
              <tr>
                {['文件名', '大小', '状态', '上传时间', '操作'].map(h => (
                  <th key={h} className="text-left px-4 py-2 text-xs font-medium text-ink-soft">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {kb.files.map(f => (
                <tr key={f.id} className="hover:bg-canvas">
                  <td className="px-4 py-2 text-ink max-w-xs truncate" title={f.original_name}>
                    {f.original_name}
                  </td>
                  <td className="px-4 py-2 text-ink-soft">{formatBytes(f.file_size)}</td>
                  <td className="px-4 py-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      f.status === 'indexed' ? 'bg-ok-bg text-ok-fg'
                      : f.status === 'error' ? 'bg-danger-bg text-danger-fg'
                      : 'bg-surface-2 text-ink-soft'
                    }`}>
                      {f.status === 'indexed' ? '已索引' : f.status === 'error' ? '错误' : '已上传'}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-muted">{formatTime(f.created_at)}</td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() => requestConfirm({
                        action: () => onDeleteFile(kb.course_id, f.id),
                        title: '删除文件',
                        message: `确认删除文件「${f.original_name}」？删除后需要重新构建索引。`,
                        confirmLabel: '确认删除',
                        variant: 'danger',
                      })}
                      className="text-xs text-danger-fg hover:text-danger-fg"
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
