import React, { useCallback, useEffect, useState } from 'react'
import {
  deleteMySearchConfig,
  fetchMySearchConfig,
  fetchSearchProviders,
  probeSearchConfig,
  putMySearchConfig,
  type SearchConfigPayload,
  type SearchProbeResult,
  type SearchProviderInfo,
} from '../../services/api'

const EMPTY: SearchConfigPayload = { provider: '', api_key: '', base_url: '', max_results: 0, proxy: '' }

const inputCls =
  'w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-600 mb-1">{label}</label>
      {children}
    </div>
  )
}

export default function UserSearchSettingsPage({ onBack }: { onBack: () => void }) {
  const [providers, setProviders] = useState<SearchProviderInfo[]>([])
  const [draft, setDraft] = useState<SearchConfigPayload>(EMPTY)
  const [hasOverride, setHasOverride] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<SearchProbeResult | null>(null)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const [p, cfg] = await Promise.all([fetchSearchProviders(), fetchMySearchConfig()])
      setProviders(p)
      setDraft({
        provider: cfg.provider,
        api_key: cfg.api_key,
        base_url: cfg.base_url,
        max_results: cfg.max_results,
        proxy: cfg.proxy,
      })
      setHasOverride(cfg.has_override)
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

  const handleSave = async () => {
    setSaving(true)
    setError('')
    setSaved(false)
    try {
      await putMySearchConfig(draft)
      setHasOverride(true)
      setSaved(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    if (!draft.provider) {
      setError('请先选择搜索引擎（或使用管理员默认则无需在此测试）')
      return
    }
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

  const handleReset = async () => {
    setSaving(true)
    setError('')
    try {
      await deleteMySearchConfig()
      setDraft(EMPTY)
      setHasOverride(false)
      setTestResult(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '恢复失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200">
        <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">← 返回</button>
        <h1 className="text-lg font-semibold text-slate-800 mt-1">🔍 我的搜索设置</h1>
        <p className="text-xs text-slate-400 mt-1">
          用你自己的 API Key 覆盖管理员默认；留空的字段会回退到管理员默认配置。
        </p>
      </header>
      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <p className="text-sm text-slate-400">加载中…</p>
        ) : (
          <div className="max-w-xl bg-white rounded-xl border border-slate-200 p-5 space-y-3">
            <div className="px-3 py-2 text-xs rounded bg-slate-50 text-slate-500">
              {hasOverride ? '你已设置自定义搜索配置（覆盖管理员默认）' : '当前使用管理员默认配置，可在下方自定义覆盖'}
            </div>
            {testResult && (
              <div
                className={`px-3 py-2 text-xs rounded ${testResult.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}
              >
                {testResult.ok ? `✓ 连通成功（${testResult.provider}）` : `✗ ${testResult.error}`}
              </div>
            )}
            {error && <div className="px-3 py-2 text-xs rounded bg-red-50 text-red-600">{error}</div>}
            {saved && <div className="px-3 py-2 text-xs rounded bg-green-50 text-green-700">✓ 已保存</div>}

            <Field label="搜索引擎（留空 = 用管理员默认）">
              <select
                value={draft.provider}
                onChange={(e) => setDraft({ ...draft, provider: e.target.value })}
                className={inputCls}
              >
                <option value="">— 用管理员默认 —</option>
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                    {p.requires_api_key ? '（需 API Key）' : ''}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="API Key（你自己的 key；留空 = 用管理员默认）">
              <input
                value={draft.api_key}
                onChange={(e) => setDraft({ ...draft, api_key: e.target.value })}
                className={inputCls}
                placeholder="留空用管理员默认"
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
            <Field label="最大结果数（0 = 用管理员默认）">
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
                className="px-3 py-1.5 text-sm rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              >
                {testing ? '测试中…' : '测试连通'}
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-sm rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? '保存中…' : '保存'}
              </button>
              {hasOverride && (
                <button
                  onClick={handleReset}
                  disabled={saving}
                  className="px-3 py-1.5 text-sm rounded-lg border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-50"
                >
                  恢复默认
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
