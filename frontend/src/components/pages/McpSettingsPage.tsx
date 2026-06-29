import { useEffect, useState, useCallback } from 'react'
import {
  fetchMcpServers,
  upsertMcpServer,
  deleteMcpServer,
  testMcpProbe,
  fetchMcpCatalog,
  fetchMyMcpEnabled,
  setMyMcpEnabled,
  type McpServerConfig,
  type McpServersResponse,
  type McpProbeResult,
  type McpCatalogServer,
} from '../../services/api'
import { getUser } from '../../services/auth'
import { Modal, Badge, Toggle, Card, EmptyState, StatusDot } from '../ui'

interface Props {
  onBack: () => void
}

const EMPTY_CONFIG: McpServerConfig = {
  type: null,
  command: '',
  args: [],
  env: {},
  cwd: '',
  url: '',
  headers: {},
  tool_timeout: 30,
  enabled_tools: ['*'],
  enabled: true,
}

export default function McpSettingsPage({ onBack }: Props) {
  const isAdmin = getUser()?.role === 'admin'
  // server 进程系统级共享，仅 admin 可配置；其余角色只读目录 + 勾选个人启用
  return isAdmin ? <McpAdminView onBack={onBack} /> : <McpStudentView onBack={onBack} />
}

// ── 学生 / 教师视图：只读目录 + 个人启用开关 ──────────────────────────────

function McpStudentView({ onBack }: { onBack: () => void }) {
  const [servers, setServers] = useState<McpCatalogServer[]>([])
  const [enabled, setEnabled] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [catalog, en] = await Promise.all([fetchMcpCatalog(), fetchMyMcpEnabled()])
      setServers(catalog)
      setEnabled(en)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const toggle = async (name: string) => {
    const next = enabled.includes(name) ? enabled.filter((n) => n !== name) : [...enabled, name]
    const prev = enabled
    setEnabled(next)
    try {
      const saved = await setMyMcpEnabled(next)
      setEnabled(saved)
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存失败')
      setEnabled(prev)
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">
            ← 返回
          </button>
          <h1 className="text-lg font-semibold text-slate-800">🔌 MCP 工具</h1>
          <Badge color="indigo">{servers.length}</Badge>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <p className="text-xs text-slate-400 mb-4">
          勾选你想在对话中启用的 MCP 工具。未勾选的对你不可见；server 由管理员统一部署，进程共享。
          未勾选任何项时默认全部可用（除非你显式自定义）。
        </p>
        {error && (
          <div className="mb-4 px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>
        )}
        {loading ? (
          <div className="text-center text-slate-400 py-16">加载中...</div>
        ) : servers.length === 0 ? (
          <EmptyState icon="🔌" title="管理员尚未配置 MCP server" hint="可用的 MCP 工具会在这里列出" />
        ) : (
          <div className="space-y-3">
            {servers.map((s) => (
              <Card key={s.name} className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <StatusDot status={s.connected ? 'connected' : 'disabled'} />
                      <h3 className="font-mono text-sm font-semibold text-slate-800">{s.name}</h3>
                      <Badge color="blue">{s.transport}</Badge>
                      {s.connected ? (
                        <Badge color="green">已连接</Badge>
                      ) : (
                        <Badge color="slate">未连接</Badge>
                      )}
                    </div>
                    {s.tools.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {s.tools.map((t) => (
                          <span
                            key={t.name}
                            className="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded font-mono"
                            title={t.description}
                          >
                            {t.name}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <label className="flex items-center gap-2 text-xs text-slate-500 cursor-pointer shrink-0">
                    <Toggle checked={enabled.includes(s.name)} onChange={() => void toggle(s.name)} />
                    启用
                  </label>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── 管理员视图：完整 server 配置（增删改 + 测试）──────────────────────────

function McpAdminView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<McpServersResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState<{ name: string; isNew: boolean } | null>(null)
  const [draftName, setDraftName] = useState('')
  const [draft, setDraft] = useState<McpServerConfig>(EMPTY_CONFIG)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<McpProbeResult | null>(null)
  const [saving, setSaving] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      setData(await fetchMcpServers())
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const openNew = () => {
    setEditing({ name: '', isNew: true })
    setDraftName('')
    setDraft(EMPTY_CONFIG)
    setTestResult(null)
  }
  const openEdit = (name: string) => {
    const cfg = data?.config[name]
    if (!cfg) return
    setEditing({ name, isNew: false })
    setDraftName(name)
    setDraft({ ...cfg })
    setTestResult(null)
  }

  const handleSave = async () => {
    const name = editing?.isNew ? draftName.trim() : editing?.name
    if (!name) {
      setError('server 名不能为空')
      return
    }
    setSaving(true)
    setError('')
    try {
      await upsertMcpServer(name, draft)
      setEditing(null)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    setError('')
    try {
      setTestResult(await testMcpProbe(draft))
    } catch (e) {
      setError(e instanceof Error ? e.message : '测试失败')
    } finally {
      setTesting(false)
    }
  }

  const handleDelete = async (name: string) => {
    if (!confirm(`确认删除 server「${name}」？`)) return
    try {
      await deleteMcpServer(name)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const transport =
    draft.type ||
    (draft.command
      ? 'stdio'
      : draft.url
        ? draft.url.endsWith('/sse')
          ? 'sse'
          : 'streamableHttp'
        : '(未配置)')

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">
            ← 返回
          </button>
          <h1 className="text-lg font-semibold text-slate-800">🔌 MCP 服务器配置</h1>
          <Badge color="indigo">{data?.servers.length ?? 0}</Badge>
        </div>
        <button
          onClick={openNew}
          className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
        >
          + 添加 server
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <p className="text-xs text-slate-400 mb-4">
          MCP（Model Context Protocol）工具以 <code>mcp_&lt;server&gt;_&lt;tool&gt;</code> 形式经
          <code> load_tools </code>渐进式揭示挂载到 agent loop。部署级配置（全局共享）。
        </p>
        {error && (
          <div className="mb-4 px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>
        )}
        {loading ? (
          <div className="text-center text-slate-400 py-16">加载中...</div>
        ) : !data || data.servers.length === 0 ? (
          <EmptyState icon="🔌" title="还没有配置 MCP server" hint="点击右上角添加（stdio / sse / streamableHttp）" />
        ) : (
          <div className="space-y-3">
            {data.servers.map((s) => {
              const cfg = data.config[s.name]
              return (
                <Card key={s.name} className="p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <StatusDot status={s.status} />
                        <h3 className="font-mono text-sm font-semibold text-slate-800">{s.name}</h3>
                        <Badge color="blue">{s.transport}</Badge>
                        {s.status === 'connected' ? (
                          <Badge color="green">已连接</Badge>
                        ) : s.status === 'error' ? (
                          <Badge color="red">错误</Badge>
                        ) : (
                          <Badge color="amber">{s.status}</Badge>
                        )}
                        {cfg && !cfg.enabled && <Badge color="slate">已禁用</Badge>}
                      </div>
                      <p className="text-xs text-slate-500 font-mono truncate">
                        {cfg?.command
                          ? `${cfg.command} ${(cfg.args || []).join(' ')}`
                          : cfg?.url || ''}
                      </p>
                      {s.error && <p className="text-xs text-red-500 mt-1">{s.error}</p>}
                      {s.tools.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {s.tools.map((t) => (
                            <span
                              key={t.name}
                              className="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded font-mono"
                              title={t.description}
                            >
                              {t.name}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <button
                        onClick={() => openEdit(s.name)}
                        className="text-xs px-2 py-1 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded"
                      >
                        编辑
                      </button>
                      <button
                        onClick={() => handleDelete(s.name)}
                        className="text-xs px-2 py-1 text-slate-500 hover:text-red-500 hover:bg-red-50 rounded"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </Card>
              )
            })}
          </div>
        )}
      </div>

      {editing && (
        <Modal
          open
          onClose={() => setEditing(null)}
          overlayCloses={false}
          title={editing.isNew ? '添加 MCP server' : `编辑 ${editing.name}`}
          footer={
            <>
              <button
                onClick={handleTest}
                disabled={testing}
                className="px-3 py-1.5 text-sm text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 disabled:opacity-50"
              >
                {testing ? '测试中...' : '测试连接'}
              </button>
              <button
                onClick={() => setEditing(null)}
                className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700"
              >
                取消
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? '保存中...' : '保存'}
              </button>
            </>
          }
        >
          <div className="space-y-3">
            {testResult && (
              <div
                className={`px-3 py-2 text-xs rounded ${
                  testResult.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'
                }`}
              >
                {testResult.ok
                  ? `✓ 连接成功，检测到 ${testResult.tools.length} 个工具`
                  : `✗ ${testResult.error}`}
                {testResult.ok && testResult.tools.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {testResult.tools.map((t) => (
                      <span key={t.name} className="font-mono px-1 bg-white rounded">
                        {t.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {editing.isNew && (
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">名称</label>
                <input
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
                  placeholder="如 math-tools"
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">
                传输方式（当前：{transport}）
              </label>
              <select
                value={draft.type || ''}
                onChange={(e) => setDraft({ ...draft, type: e.target.value || null })}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
              >
                <option value="">自动检测（推荐）</option>
                <option value="stdio">stdio（本地进程）</option>
                <option value="sse">sse</option>
                <option value="streamableHttp">streamableHttp</option>
              </select>
            </div>
            {(!draft.type || draft.type === 'stdio') && (
              <>
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">命令（command）</label>
                  <input
                    value={draft.command}
                    onChange={(e) => setDraft({ ...draft, command: e.target.value })}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
                    placeholder="npx"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">参数（每行一个）</label>
                  <textarea
                    value={(draft.args || []).join('\n')}
                    onChange={(e) =>
                      setDraft({ ...draft, args: e.target.value.split('\n').filter(Boolean) })
                    }
                    rows={3}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
                    placeholder={'-y\n@modelcontextprotocol/server-filesystem'}
                  />
                </div>
              </>
            )}
            {(!draft.type || draft.type !== 'stdio') && (
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">URL</label>
                <input
                  value={draft.url}
                  onChange={(e) => setDraft({ ...draft, url: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
                  placeholder="https://example.com/mcp"
                />
              </div>
            )}
            <KeyValueEditor
              title="环境变量（env）"
              value={draft.env}
              onChange={(env) => setDraft({ ...draft, env })}
            />
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">工具超时（秒）</label>
              <input
                type="number"
                value={draft.tool_timeout}
                onChange={(e) =>
                  setDraft({ ...draft, tool_timeout: Number(e.target.value) || 30 })
                }
                className="w-32 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
              <Toggle
                checked={draft.enabled}
                onChange={(v) => setDraft({ ...draft, enabled: v })}
              />{' '}
              启用
            </label>
          </div>
        </Modal>
      )}
    </div>
  )
}

function KeyValueEditor({
  title,
  value,
  onChange,
}: {
  title: string
  value: Record<string, string>
  onChange: (v: Record<string, string>) => void
}) {
  const entries = Object.entries(value)
  return (
    <div>
      <label className="block text-xs font-medium text-slate-600 mb-1">{title}</label>
      <div className="space-y-1.5">
        {entries.map(([k, v], i) => (
          <div key={i} className="flex gap-1.5">
            <input
              value={k}
              onChange={(e) => {
                const next = { ...value }
                delete next[k]
                next[e.target.value] = v
                onChange(next)
              }}
              className="flex-1 rounded border border-slate-200 px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-400"
              placeholder="KEY"
            />
            <input
              value={v}
              onChange={(e) => onChange({ ...value, [k]: e.target.value })}
              className="flex-1 rounded border border-slate-200 px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-400"
              placeholder="value"
            />
            <button
              onClick={() => {
                const next = { ...value }
                delete next[k]
                onChange(next)
              }}
              className="text-xs text-slate-400 hover:text-red-500 px-1"
            >
              ✕
            </button>
          </div>
        ))}
        <button
          onClick={() => onChange({ ...value, '': '' })}
          className="text-xs text-indigo-600 hover:text-indigo-800"
        >
          + 添加
        </button>
      </div>
    </div>
  )
}
