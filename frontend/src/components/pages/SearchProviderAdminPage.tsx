import React, { useCallback, useEffect, useState } from 'react'
import { getUser } from '../../services/auth'
import {
  fetchSearchAdminConfig,
  fetchSearchProviders,
  probeSearchConfig,
  putSearchAdminConfig,
  type SearchConfigPayload,
  type SearchProbeResult,
  type SearchProviderInfo,
} from '../../services/api'

const EMPTY: SearchConfigPayload = { provider: 'duckduckgo', api_key: '', base_url: '', max_results: 5, proxy: '' }

const inputCls =
  'w-full rounded-[var(--radius)] border border-line px-3 py-2 text-sm font-mono focus:outline-none focus:border-ink'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-ink-soft mb-1">{label}</label>
      {children}
    </div>
  )
}

export default function SearchProviderAdminPage({ onBack }: { onBack: () => void }) {
  const isAdmin = getUser()?.role === 'admin'
  const [providers, setProviders] = useState<SearchProviderInfo[]>([])
  const [draft, setDraft] = useState<SearchConfigPayload>(EMPTY)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<SearchProbeResult | null>(null)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const [p, cfg] = await Promise.all([fetchSearchProviders(), fetchSearchAdminConfig()])
      setProviders(p)
      setDraft({ ...EMPTY, ...cfg })
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

  if (!isAdmin) {
    return (
      <div className="h-full flex flex-col bg-canvas">
        <header className="px-6 py-4 bg-surface border-b border-line">
          <button onClick={onBack} className="text-muted hover:text-ink-soft text-sm">← 返回</button>
        </header>
        <div className="flex-1 flex flex-col items-center justify-center text-muted gap-2">
          <span className="text-4xl"></span>
          <p className="text-sm">搜索引擎默认配置由管理员统一管理</p>
        </div>
      </div>
    )
  }

  const handleSave = async () => {
    setSaving(true)
    setError('')
    setSaved(false)
    try {
      await putSearchAdminConfig(draft)
      setSaved(true)
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
      setTestResult(
        await probeSearchConfig({ provider: draft.provider, api_key: draft.api_key, base_url: draft.base_url }),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : '测试失败')
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="px-6 py-4 bg-surface border-b border-line">
        <button onClick={onBack} className="text-muted hover:text-ink-soft text-sm">← 返回</button>
        <h1 className="text-lg font-semibold text-ink mt-1">联网搜索引擎（全局默认）</h1>
        <p className="text-xs text-muted mt-1">
          管理员配置的默认搜索引擎；普通用户可在「我的搜索设置」用各自的 key 覆盖。
        </p>
      </header>
      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <p className="text-sm text-muted">加载中…</p>
        ) : (
          <div className="max-w-xl bg-surface rounded-[var(--radius)] border border-line p-5 space-y-3">
            {testResult && (
              <div
                className={`px-3 py-2 text-xs rounded ${testResult.ok ? 'bg-ok-bg text-ok-fg' : 'bg-danger-bg text-danger-fg'}`}
              >
                {testResult.ok ? `✓ 连通成功（${testResult.provider}）` : `✗ ${testResult.error}`}
              </div>
            )}
            {error && <div className="px-3 py-2 text-xs rounded bg-danger-bg text-danger-fg">{error}</div>}
            {saved && <div className="px-3 py-2 text-xs rounded bg-ok-bg text-ok-fg">✓ 已保存</div>}

            <Field label="搜索引擎（provider）">
              <select
                value={draft.provider}
                onChange={(e) => setDraft({ ...draft, provider: e.target.value })}
                className={inputCls}
              >
                <option value="">— 选择 —</option>
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                    {p.requires_api_key ? '（需 API Key）' : ''}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="API Key（duckduckgo 无需；留空走 .env）">
              <input
                value={draft.api_key}
                onChange={(e) => setDraft({ ...draft, api_key: e.target.value })}
                className={inputCls}
                placeholder="留空走 .env"
              />
            </Field>
            <Field label="Base URL（searxng 必填；其他通常留空）">
              <input
                value={draft.base_url}
                onChange={(e) => setDraft({ ...draft, base_url: e.target.value })}
                className={inputCls}
                placeholder="https://..."
              />
            </Field>
            <Field label="最大结果数">
              <input
                type="number"
                value={draft.max_results}
                onChange={(e) => setDraft({ ...draft, max_results: Number(e.target.value) })}
                className={inputCls}
              />
            </Field>
            <Field label="代理（可选）">
              <input
                value={draft.proxy}
                onChange={(e) => setDraft({ ...draft, proxy: e.target.value })}
                className={inputCls}
                placeholder="http://proxy:8080"
              />
            </Field>
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleTest}
                disabled={testing}
                className="px-3 py-1.5 text-sm rounded-[var(--radius)] border border-line text-ink-soft hover:bg-canvas disabled:opacity-50"
              >
                {testing ? '测试中…' : '测试连通'}
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-sm rounded-[var(--radius)] bg-accent text-white hover:bg-accent-2 disabled:opacity-50"
              >
                {saving ? '保存中…' : '保存'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
