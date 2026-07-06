import { useEffect, useState, useCallback } from 'react'
import { Sparkles, Plus } from 'lucide-react'
import {
  fetchSkillKnowledge,
  fetchSkillDetail,
  createSkillKnowledge,
  updateSkillKnowledge,
  deleteSkillKnowledge,
  type SkillSummaryEntry,
  type SkillKnowledgeDetail,
} from '../../services/api'
import { Modal, Badge, Toggle, Card, EmptyState, Button } from '../ui'
import { getUser } from '../../services/auth'

interface Props {
  courseId: string
  onBack: () => void
}

interface EditorPayload {
  name?: string
  description?: string
  content?: string
  always?: boolean
  rename_to?: string
}

export default function SkillKnowledgePage({ courseId, onBack }: Props) {
  const user = getUser()
  const isStudent = user?.role === 'student'
  const [skills, setSkills] = useState<SkillSummaryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState<SkillKnowledgeDetail | null>(null)
  const [creating, setCreating] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      setSkills(await fetchSkillKnowledge(courseId))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [courseId])

  useEffect(() => {
    void reload()
  }, [reload])

  const openEdit = async (name: string) => {
    try {
      setEditing(await fetchSkillDetail(name, courseId))
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载详情失败')
    }
  }

  const handleDelete = async (name: string) => {
    if (!confirm(`确认删除技能「${name}」？`)) return
    try {
      await deleteSkillKnowledge(name, courseId)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleToggleAlways = async (s: SkillSummaryEntry) => {
    try {
      await updateSkillKnowledge(s.name, { always: !s.always, course_id: courseId })
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '更新失败')
    }
  }

  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="px-6 py-4 bg-surface border-b border-line flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-muted hover:text-ink text-sm transition">
            ← 返回
          </button>
          <h1 className="text-lg font-serif text-ink inline-flex items-center gap-2">
            <Sparkles size={18} strokeWidth={1.5} className="text-ink-soft" />
            {isStudent ? '我的技能包' : '课程技能包'}
          </h1>
          <Badge color="neutral">{skills.length}</Badge>
        </div>
        <Button variant="primary" size="sm" icon={Plus} onClick={() => setCreating(true)}>
          新建技能
        </Button>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <p className="text-xs text-muted mb-4 leading-relaxed">
          {isStudent
            ? '你的私人技能包（SKILL.md playbook）：在你与本课程的对话中，模型会在任务匹配时按需读取，作为给你的专项指引。仅你可见，不影响其他同学。'
            : '课程技能包（SKILL.md playbook）：模型在任务匹配时经 read_skill 按需读取，用于教学场景的专项指引（如「概念分层讲解」「错因诊断」）。同课学生共享。区别于对话后补充框（Output Cards）。'}
        </p>
        {error && (
          <div className="mb-4 px-4 py-2 bg-danger-bg text-danger-fg text-sm rounded-[var(--radius)]">{error}</div>
        )}
        {loading ? (
          <div className="text-center text-muted py-16">加载中...</div>
        ) : skills.length === 0 ? (
          <EmptyState icon={Sparkles} title="还没有技能知识包" hint="点击右上角「新建技能」创建教学 playbook" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {skills.map((s) => (
              <Card key={s.name} className="p-4">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-mono text-sm font-semibold text-ink truncate">{s.name}</h3>
                  {s.source === 'builtin' ? (
                    <Badge color="neutral">内置</Badge>
                  ) : s.source === 'personal' ? (
                    <Badge color="ok">个人</Badge>
                  ) : (
                    <Badge color="info">课程</Badge>
                  )}
                  {s.always && <Badge color="warn">常驻</Badge>}
                </div>
                <p className="text-xs text-muted line-clamp-2 mb-3">
                  {s.description || '（无描述）'}
                </p>
                <div className="flex items-center justify-between pt-3 border-t border-line">
                  <label className="flex items-center gap-1.5 text-xs text-muted cursor-pointer">
                    <Toggle checked={s.always} onChange={() => handleToggleAlways(s)} /> 常驻注入
                  </label>
                  <div className="flex gap-1">
                    <button
                      onClick={() => openEdit(s.name)}
                      className="text-xs px-2 py-1 text-ink-soft hover:text-ink hover:bg-surface-2 rounded-[var(--radius-sm)] transition"
                    >
                      查看/编辑
                    </button>
                    {s.source !== 'builtin' && (
                      <button
                        onClick={() => handleDelete(s.name)}
                        className="text-xs px-2 py-1 text-ink-soft hover:text-danger-fg hover:bg-danger-bg rounded-[var(--radius-sm)] transition"
                      >
                        删除
                      </button>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {(editing || creating) && (
        <SkillEditor
          detail={editing}
          onClose={() => {
            setEditing(null)
            setCreating(false)
          }}
          onSave={async (payload) => {
            if (editing) {
              await updateSkillKnowledge(editing.name, { ...payload, course_id: courseId })
            } else {
              await createSkillKnowledge({
                name: payload.name!,
                description: payload.description!,
                content: payload.content || '',
                always: payload.always,
                course_id: courseId,
              })
            }
            setEditing(null)
            setCreating(false)
            void reload()
          }}
        />
      )}
    </div>
  )
}

function SkillEditor({
  detail,
  onClose,
  onSave,
}: {
  detail: SkillKnowledgeDetail | null
  onClose: () => void
  onSave: (payload: EditorPayload) => Promise<void>
}) {
  const [name, setName] = useState(detail?.name ?? '')
  const [description, setDescription] = useState(detail?.description ?? '')
  const [content, setContent] = useState(detail?.content ?? '')
  const [always, setAlways] = useState(detail?.always ?? false)
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')

  const readOnly = detail?.read_only ?? false

  const submit = async () => {
    if (!detail && !name.trim()) {
      setErr('name 不能为空')
      return
    }
    if (!description.trim()) {
      setErr('description 不能为空')
      return
    }
    setSaving(true)
    setErr('')
    try {
      await onSave({
        name: name.trim(),
        description: description.trim(),
        content,
        always,
        rename_to: detail && detail.name !== name.trim() ? name.trim() : undefined,
      })
    } catch (e) {
      setErr(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const inputCls = 'w-full rounded-[var(--radius)] border border-line px-3 py-2 text-sm bg-surface text-ink placeholder:text-muted focus:outline-none focus:border-ink disabled:bg-surface-2 disabled:text-muted'

  return (
    <Modal
      open
      onClose={onClose}
      overlayCloses={false}
      title={detail ? (readOnly ? `查看 ${detail.name}` : `编辑 ${detail.name}`) : '新建技能'}
      footer={
        <>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-ink-soft hover:text-ink transition"
          >
            取消
          </button>
          {!readOnly && (
            <button
              onClick={submit}
              disabled={saving}
              className="px-3 py-1.5 text-sm bg-accent hover:bg-accent-2 text-white rounded-[var(--radius)] disabled:opacity-50 transition"
            >
              {saving ? '保存中...' : '保存'}
            </button>
          )}
        </>
      }
    >
      <div className="space-y-3">
        {err && <div className="px-3 py-2 bg-danger-bg text-danger-fg text-xs rounded-[var(--radius-sm)]">{err}</div>}
        <div>
          <label className="block text-xs font-medium text-muted mb-1">
            技能名（小写字母/数字/连字符）
          </label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value.toLowerCase())}
            disabled={readOnly}
            className={`${inputCls} font-mono`}
            placeholder="如 concept-explain"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-muted mb-1">
            描述（任务匹配时触发，写清何时用）
          </label>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={readOnly}
            className={inputCls}
            placeholder="如：分层讲解一个概念。当学生问「X 是什么」时使用。"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-muted mb-1">
            正文（Markdown playbook，模型按此执行）
          </label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            disabled={readOnly}
            rows={12}
            className={`${inputCls} font-mono`}
            placeholder={'# 技能标题\n\n按以下步骤...'}
          />
        </div>
        <label className="flex items-center gap-2 text-sm text-ink-soft cursor-pointer">
          <Toggle checked={always} onChange={setAlways} disabled={readOnly} />
          常驻注入（always: true，每轮都注入，不进 read_skill 清单）
        </label>
      </div>
    </Modal>
  )
}
