import { useEffect, useState, useCallback } from 'react'
import {
  fetchNotebookEntries,
  updateNotebookEntry,
  deleteNotebookEntry,
  fetchNotebookCategories,
  createNotebookCategory,
  deleteNotebookCategory,
  type NotebookEntry,
  type NotebookCategory,
} from '../../services/api'
import { Badge, Toggle, Card, EmptyState } from '../ui'
import { NotebookPen } from 'lucide-react'

interface Props {
  onBack: () => void
}

type Filter = 'all' | 'bookmarked' | 'wrong'

export default function NotebookPage({ onBack }: Props) {
  const [entries, setEntries] = useState<NotebookEntry[]>([])
  const [categories, setCategories] = useState<NotebookCategory[]>([])
  const [filter, setFilter] = useState<Filter>('all')
  const [activeCategory, setActiveCategory] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params: {
        category_id?: number
        bookmarked?: boolean
        is_correct?: boolean
        limit?: number
      } = { limit: 100 }
      if (filter === 'bookmarked') params.bookmarked = true
      if (filter === 'wrong') params.is_correct = false
      if (activeCategory != null) params.category_id = activeCategory
      const [entryData, cats] = await Promise.all([
        fetchNotebookEntries(params),
        fetchNotebookCategories(),
      ])
      setEntries(entryData.items)
      setCategories(cats)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [filter, activeCategory])

  useEffect(() => {
    void reload()
  }, [reload])

  const toggleBookmark = async (e: NotebookEntry) => {
    try {
      await updateNotebookEntry(e.id, { bookmarked: !e.bookmarked })
      void reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '操作失败')
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('确认删除这道题？')) return
    try {
      await deleteNotebookEntry(id)
      void reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除失败')
    }
  }

  const handleAddCategory = async () => {
    const name = prompt('分类名称')
    if (!name) return
    try {
      await createNotebookCategory(name)
      void reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建失败')
    }
  }

  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="px-6 py-4 bg-surface border-b border-line flex items-center gap-3">
        <button onClick={onBack} className="text-muted hover:text-ink-soft text-sm">
          ← 返回
        </button>
        <h1 className="text-lg font-semibold text-ink">题目笔记本</h1>
        <Badge color="indigo">{entries.length}</Badge>
      </header>
      <div className="flex-1 overflow-hidden flex">
        <div className="w-56 border-r border-line bg-surface p-3 overflow-y-auto shrink-0">
          <span className="block text-xs font-medium text-ink-soft mb-2">过滤</span>
          {(['all', 'bookmarked', 'wrong'] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => {
                setFilter(f)
                setActiveCategory(null)
              }}
              className={`w-full text-left text-sm px-2 py-1.5 rounded mb-0.5 ${
                filter === f && activeCategory == null
                  ? 'bg-surface-2 text-ink'
                  : 'text-ink-soft hover:bg-canvas'
              }`}
            >
              {f === 'all' ? '全部' : f === 'bookmarked' ? '⭐ 已收藏' : '✗ 错题'}
            </button>
          ))}
          <div className="flex items-center justify-between mt-4 mb-2">
            <span className="text-xs font-medium text-ink-soft">分类</span>
            <button
              onClick={handleAddCategory}
              className="text-xs text-ink hover:text-ink"
            >
              +
            </button>
          </div>
          {categories.map((c) => (
            <div key={c.id} className="flex items-center group">
              <button
                onClick={() => {
                  setActiveCategory(c.id)
                  setFilter('all')
                }}
                className={`flex-1 text-left text-sm px-2 py-1.5 rounded ${
                  activeCategory === c.id
                    ? 'bg-surface-2 text-ink'
                    : 'text-ink-soft hover:bg-canvas'
                }`}
              >
                {c.name} <span className="text-xs text-muted">({c.entry_count})</span>
              </button>
              <button
                onClick={() => {
                  if (confirm(`删除分类「${c.name}」？`)) {
                    void deleteNotebookCategory(c.id).then(() => void reload())
                  }
                }}
                className="opacity-0 group-hover:opacity-100 text-xs text-muted hover:text-danger-fg px-1"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          {error && (
            <div className="mb-4 px-4 py-2 bg-danger-bg text-danger-fg text-sm rounded-[var(--radius)]">{error}</div>
          )}
          {loading ? (
            <div className="text-center text-muted py-16">加载中...</div>
          ) : entries.length === 0 ? (
            <EmptyState icon={NotebookPen} title="笔记本还是空的" hint="出题/答题后会自动收录，或手动收藏" />
          ) : (
            <div className="space-y-3">
              {entries.map((e) => (
                <Card key={e.id} className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    {e.difficulty && (
                      <Badge
                        color={
                          e.difficulty === 'hard'
                            ? 'red'
                            : e.difficulty === 'easy'
                              ? 'green'
                              : 'amber'
                        }
                      >
                        {e.difficulty}
                      </Badge>
                    )}
                    {e.question_type && <Badge color="blue">{e.question_type}</Badge>}
                    {e.is_correct ? (
                      <Badge color="green">✓ 正确</Badge>
                    ) : (
                      <Badge color="red">✗ 错误</Badge>
                    )}
                    <span className="ml-auto text-xs text-muted">
                      {new Date(e.created_at * 1000).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-sm text-ink mb-2 font-medium">{e.question}</p>
                  {e.options && Object.keys(e.options).length > 0 && (
                    <div className="text-xs text-ink-soft space-y-0.5 mb-2">
                      {Object.entries(e.options).map(([k, v]) => (
                        <div
                          key={k}
                          className={
                            k === e.correct_answer
                              ? 'text-ok-fg font-medium'
                              : k === e.user_answer && !e.is_correct
                                ? 'text-danger-fg'
                                : ''
                          }
                        >
                          {k}. {v}
                          {k === e.correct_answer
                            ? ' ✓'
                            : k === e.user_answer && !e.is_correct
                              ? ' ✗你的'
                              : ''}
                        </div>
                      ))}
                    </div>
                  )}
                  {e.correct_answer && !e.options?.[e.correct_answer] && (
                    <p className="text-xs text-ink-soft mb-1">
                      答案：<span className="text-ok-fg font-medium">{e.correct_answer}</span>
                    </p>
                  )}
                  {e.explanation && (
                    <p className="text-xs text-ink-soft mt-2 pt-2 border-t border-line">
                      {e.explanation}
                    </p>
                  )}
                  <div className="flex items-center justify-end gap-2 mt-2">
                    <label className="flex items-center gap-1 text-xs text-ink-soft cursor-pointer">
                      <Toggle checked={e.bookmarked} onChange={() => toggleBookmark(e)} /> 收藏
                    </label>
                    <button
                      onClick={() => handleDelete(e.id)}
                      className="text-xs text-muted hover:text-danger-fg"
                    >
                      删除
                    </button>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
