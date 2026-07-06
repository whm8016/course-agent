import { useEffect, useState, useCallback } from 'react'
import {
  fetchLlmProfilesAdmin,
  upsertLlmProfile,
  deleteLlmProfile,
  setActiveLlmProfile,
  testLlmProfile,
  probeLlmProfile,
  fetchLlmProviders,
  fetchMyLlmProvider,
  upsertMyLlmProvider,
  deleteMyLlmProvider,
  testMyLlmProvider,
  type LlmProfileAdmin,
  type LlmProviderSpec,
  type LlmProbeResult,
  type UserProviderPayload,
} from '../../services/api'
import { getUser } from '../../services/auth'
import { Modal, Badge, Card, EmptyState } from '../ui'

interface Props {
  onBack: () => void
}

const EMPTY_DRAFT: LlmProfileAdmin = {
  id: '',
  name: '',
  binding: '',
  api_key: '',
  base_url: '',
  api_version: '',
  text_model: '',
  fast_model: '',
  vision_model: '',
  embedding_model: '',
  embedding_api_key: '',
  embedding_base_url: '',
  fallback_api_key: '',
  fallback_base_url: '',
  fallback_model: '',
  active: false,
}

const inputCls =
  'w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400'

export default function LlmProviderPage({ onBack }: Props) {
  const isAdmin = getUser()?.role === 'admin'
  // 非 admin：学生自配个人 provider（覆盖平台默认），含视觉模型（两阶段图片描述用）
  if (!isAdmin) {
    return <LlmUserView onBack={onBack} />
  }
  return <LlmAdminView onBack={onBack} />
}

function LlmAdminView({ onBack }: { onBack: () => void }) {
  const [profiles, setProfiles] = useState<LlmProfileAdmin[]>([])
  const [providers, setProviders] = useState<LlmProviderSpec[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState<{ id: string; isNew: boolean } | null>(null)
  const [draftId, setDraftId] = useState('')
  const [draft, setDraft] = useState<LlmProfileAdmin>(EMPTY_DRAFT)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<LlmProbeResult | null>(null)
  const [saving, setSaving] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const [p, prov] = await Promise.all([fetchLlmProfilesAdmin(), fetchLlmProviders()])
      setProfiles(p.profiles || [])
      setProviders(prov || [])
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
    setEditing({ id: '', isNew: true })
    setDraftId('')
    setDraft({ ...EMPTY_DRAFT })
    setTestResult(null)
  }
  const openEdit = (p: LlmProfileAdmin) => {
    setEditing({ id: p.id, isNew: false })
    setDraftId(p.id)
    setDraft({ ...p })
    setTestResult(null)
  }

  const onPickBinding = (binding: string) => {
    const spec = providers.find((s) => s.name === binding)
    setDraft((d) => ({ ...d, binding, base_url: spec?.default_api_base || d.base_url }))
  }

  const handleSave = async () => {
    const id = editing?.isNew ? draftId.trim() : editing?.id
    if (!id) {
      setError('profile id 不能为空')
      return
    }
    if (!draft.text_model.trim()) {
      setError('主对话模型（text_model）不能为空')
      return
    }
    setSaving(true)
    setError('')
    try {
      await upsertLlmProfile(id, draft)
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
      setTestResult(await probeLlmProfile(draft))
    } catch (e) {
      setError(e instanceof Error ? e.message : '测试失败')
    } finally {
      setTesting(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm(`确认删除 profile「${id}」？`)) return
    try {
      await deleteLlmProfile(id)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleSetActive = async (id: string) => {
    try {
      await setActiveLlmProfile(id)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '设置失败')
    }
  }

  const handleTestSaved = async (id: string) => {
    try {
      const r = await testLlmProfile(id)
      const lines = [`💬 对话：${r.ok ? `✓ ${r.model}` : `✗ ${r.error}`}`]
      if (r.vision) {
        lines.push(`🖼️ 视觉：${r.vision.ok ? `✓ ${r.vision.model}` : `✗ ${r.vision.error}`}`)
      }
      alert(`${id}\n${lines.join('\n')}`)
    } catch (e) {
      alert(e instanceof Error ? e.message : '测试失败')
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">← 返回</button>
          <h1 className="text-lg font-semibold text-slate-800">🤖 模型供应商</h1>
          <Badge color="indigo">{profiles.length}</Badge>
        </div>
        <button
          onClick={openNew}
          className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
        >
          + 添加 profile
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <p className="text-xs text-slate-400 mb-4">
          以 <b>profile 池</b>预配多个 provider+model 组合（对标 DeepTutor）。用户在对话顶部下拉临时切换，
          后端按 profile 动态构造 client 注入 loop，<b>即时生效无需重启</b>。空字段回退 .env（active=default 通常 key/base_url 留空）。
        </p>
        {error && (
          <div className="mb-4 px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>
        )}
        {loading ? (
          <div className="text-center text-slate-400 py-16">加载中...</div>
        ) : profiles.length === 0 ? (
          <EmptyState icon="🤖" title="还没有配置模型 profile" hint="点击右上角添加（deepseek / openai / dashscope / anthropic ...）" />
        ) : (
          <div className="space-y-3">
            {profiles.map((p) => (
              <Card key={p.id} className="p-4">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <h3 className="font-mono text-sm font-semibold text-slate-800">{p.name || p.id}</h3>
                      <Badge color="blue">{p.binding || '(未设置)'}</Badge>
                      {p.active && <Badge color="green">默认</Badge>}
                      {p.api_key ? <Badge color="slate">key ✓</Badge> : <Badge color="amber">key 走 .env</Badge>}
                    </div>
                    <p className="text-xs text-slate-500 font-mono truncate">
                      text: {p.text_model || '(空)'}{p.fast_model && ` · fast: ${p.fast_model}`}
                    </p>
                    {p.base_url && (
                      <p className="text-[11px] text-slate-400 font-mono truncate">{p.base_url}</p>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1 justify-end shrink-0">
                    {!p.active && (
                      <button onClick={() => handleSetActive(p.id)} className="text-xs px-2 py-1 text-green-600 hover:bg-green-50 rounded">
                        设默认
                      </button>
                    )}
                    <button onClick={() => handleTestSaved(p.id)} className="text-xs px-2 py-1 text-indigo-600 hover:bg-indigo-50 rounded">
                      测试
                    </button>
                    <button onClick={() => openEdit(p)} className="text-xs px-2 py-1 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded">
                      编辑
                    </button>
                    <button onClick={() => handleDelete(p.id)} className="text-xs px-2 py-1 text-slate-500 hover:text-red-500 hover:bg-red-50 rounded">
                      删除
                    </button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {editing && (
        <Modal
          open
          onClose={() => setEditing(null)}
          overlayCloses={false}
          title={editing.isNew ? '添加模型 profile' : `编辑 ${editing.id}`}
          footer={
            <>
              <button
                onClick={handleTest}
                disabled={testing}
                className="px-3 py-1.5 text-sm text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 disabled:opacity-50"
              >
                {testing ? '测试中...' : '测试连接'}
              </button>
              <button onClick={() => setEditing(null)} className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700">
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
              <div className="space-y-1.5">
                <div className={`px-3 py-2 text-xs rounded ${(testResult.text?.ok ?? testResult.ok) ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                  💬 对话：{(testResult.text?.ok ?? testResult.ok)
                    ? `✓ ${testResult.text?.model ?? testResult.model}`
                    : `✗ ${testResult.text?.error ?? testResult.error}`}
                </div>
                {testResult.vision !== undefined && (
                  testResult.vision === null ? (
                    <div className="px-3 py-2 text-xs rounded bg-slate-50 text-slate-500">
                      🖼️ 视觉：未配置
                    </div>
                  ) : (
                    <div className={`px-3 py-2 text-xs rounded ${testResult.vision.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                      🖼️ 视觉：{testResult.vision.ok ? `✓ ${testResult.vision.model}` : `✗ ${testResult.vision.error}`}
                      {testResult.vision.warning && (
                        <div className="mt-0.5 text-[11px] text-amber-600">⚠ {testResult.vision.warning}</div>
                      )}
                    </div>
                  )
                )}
              </div>
            )}
            {editing.isNew && (
              <Field label="profile id（唯一标识，如 deepseek-pro / gpt4o）">
                <input value={draftId} onChange={(e) => setDraftId(e.target.value)} className={inputCls} placeholder="deepseek-pro" />
              </Field>
            )}
            <Field label="显示名称（可选）">
              <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} className={inputCls} placeholder="DeepSeek V4" />
            </Field>
            <Field label="供应商 binding（选择后自动填默认 base_url）">
              <select value={draft.binding} onChange={(e) => onPickBinding(e.target.value)} className={inputCls}>
                <option value="">— 选择 —</option>
                {providers.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                    {s.default_api_base ? `  (${s.default_api_base})` : ''}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="API Key（留空走 .env / active 默认）">
              <input value={draft.api_key} onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} className={inputCls} placeholder="sk-..." />
            </Field>
            <Field label="Base URL（留空用供应商默认端点）">
              <input value={draft.base_url} onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} className={inputCls} placeholder="https://..." />
            </Field>
            <Field label="主对话模型 text_model（必填）">
              <input value={draft.text_model} onChange={(e) => setDraft({ ...draft, text_model: e.target.value })} className={inputCls} placeholder="deepseek-chat / gpt-4o / qwen-plus" />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="快速模型 fast（可选）">
                <input value={draft.fast_model} onChange={(e) => setDraft({ ...draft, fast_model: e.target.value })} className={inputCls} />
              </Field>
              <Field label="视觉模型 vision（可选）">
                <input value={draft.vision_model} onChange={(e) => setDraft({ ...draft, vision_model: e.target.value })} className={inputCls} />
              </Field>
            </div>
            <details className="text-xs">
              <summary className="cursor-pointer text-slate-500 hover:text-slate-700">嵌入 / 兜底（高级，可选）</summary>
              <div className="space-y-2 mt-2 pl-2 border-l border-slate-200">
                <Field label="嵌入模型 embedding">
                  <input value={draft.embedding_model} onChange={(e) => setDraft({ ...draft, embedding_model: e.target.value })} className={inputCls} />
                </Field>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="嵌入 api_key（空=沿用主 key）">
                    <input value={draft.embedding_api_key} onChange={(e) => setDraft({ ...draft, embedding_api_key: e.target.value })} className={inputCls} />
                  </Field>
                  <Field label="嵌入 base_url（空=沿用主 url）">
                    <input value={draft.embedding_base_url} onChange={(e) => setDraft({ ...draft, embedding_base_url: e.target.value })} className={inputCls} />
                  </Field>
                </div>
                <Field label="兜底模型 fallback（主模型熔断时）">
                  <input value={draft.fallback_model} onChange={(e) => setDraft({ ...draft, fallback_model: e.target.value })} className={inputCls} />
                </Field>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="兜底 api_key">
                    <input value={draft.fallback_api_key} onChange={(e) => setDraft({ ...draft, fallback_api_key: e.target.value })} className={inputCls} />
                  </Field>
                  <Field label="兜底 base_url">
                    <input value={draft.fallback_base_url} onChange={(e) => setDraft({ ...draft, fallback_base_url: e.target.value })} className={inputCls} />
                  </Field>
                </div>
              </div>
            </details>
          </div>
        </Modal>
      )}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-600 mb-1">{label}</label>
      {children}
    </div>
  )
}

/** 学生自配个人 LLM provider（覆盖平台默认）。
 *
 * 含视觉模型字段：当主对话模型不支持视觉时（如 deepseek），后端用该视觉模型把图片转成
 * 文字描述再喂给主模型（两阶段，对标 ingestion 图片描述）。留空字段回退平台默认；
 * embedding 不开放（per-course 共享库要求 embedding 一致）。
 */
function LlmUserView({ onBack }: { onBack: () => void }) {
  const EMPTY: UserProviderPayload = {
    binding: '', api_key: '', base_url: '', api_version: '', text_model: '',
    vision_binding: '', vision_api_key: '', vision_base_url: '', vision_model: '',
  }
  const [providers, setProviders] = useState<LlmProviderSpec[]>([])
  const [draft, setDraft] = useState<UserProviderPayload>(EMPTY)
  const [apiKeySet, setApiKeySet] = useState(false)
  const [visionApiKeySet, setVisionApiKeySet] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<LlmProbeResult | null>(null)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [view, prov] = await Promise.all([fetchMyLlmProvider(), fetchLlmProviders()])
      setProviders(prov || [])
      setApiKeySet(!!view.api_key_set)
      setVisionApiKeySet(!!view.vision_api_key_set)
      setDraft({
        // 对话供应商
        binding: view.binding || '',
        api_key: '', // 不回传明文；留空=保留原 key
        base_url: view.base_url || '',
        api_version: view.api_version || '',
        text_model: view.text_model || '',
        // 视觉独立供应商
        vision_binding: view.vision_binding || '',
        vision_api_key: '', // 不回传明文；留空=保留原 key
        vision_base_url: view.vision_base_url || '',
        vision_model: view.vision_model || '',
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const onPickBinding = (binding: string) => {
    const spec = providers.find((s) => s.name === binding)
    setDraft((d) => ({ ...d, binding, base_url: spec?.default_api_base || d.base_url }))
  }

  const onPickVisionBinding = (binding: string) => {
    const spec = providers.find((s) => s.name === binding)
    setDraft((d) => ({ ...d, vision_binding: binding, vision_base_url: spec?.default_api_base || d.vision_base_url }))
  }

  const handleSave = async () => {
    if (!draft.binding.trim()) {
      setError('请选择供应商 binding')
      return
    }
    if (!draft.text_model.trim()) {
      setError('主对话模型 text_model 不能为空')
      return
    }
    setSaving(true)
    setError('')
    try {
      await upsertMyLlmProvider(draft)
      setTestResult(null)
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
      setTestResult(await testMyLlmProvider(draft))
    } catch (e) {
      setError(e instanceof Error ? e.message : '测试失败')
    } finally {
      setTesting(false)
    }
  }

  const handleDelete = async () => {
    if (!confirm('确认删除个人配置，回退平台默认？')) return
    try {
      await deleteMyLlmProvider()
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  const hasOverride = !!(
    draft.binding || draft.text_model ||
    draft.vision_binding || draft.vision_model
  )

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">← 返回</button>
          <h1 className="text-lg font-semibold text-slate-800">🤖 我的模型配置</h1>
          {hasOverride ? <Badge color="green">个人配置生效</Badge> : <Badge color="slate">用平台默认</Badge>}
        </div>
        {hasOverride && (
          <button onClick={handleDelete} className="text-xs px-2 py-1 text-red-500 hover:bg-red-50 rounded">
            清除个人配置
          </button>
        )}
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-4">
          <Card className="p-4 bg-indigo-50/40 border-indigo-100">
            <p className="text-xs text-slate-600 leading-relaxed">
              自配个人模型会<b>覆盖平台默认</b>，仅对你生效（即时生效，无需重启）。留空字段回退默认。
              对话模型与视觉模型<b>走各自独立的供应商</b>（如对话填 deepseek，视觉填 dashscope 的 qwen-vl），
              两把 API Key / Base URL / 模型各自独立填写。视觉模型用于：当主对话模型不支持看图时，
              由它把图片转成文字描述再交给主模型回答，这样发图片也不会再「看不到内容」。嵌入模型由平台统一（不在此开放）。
            </p>
          </Card>

          {error && <div className="px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>}
          {testResult && (
            <div className="space-y-2">
              <div className={`px-4 py-2 text-sm rounded-lg ${(testResult.text?.ok ?? testResult.ok) ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                💬 对话模型：{(testResult.text?.ok ?? testResult.ok)
                  ? `✓ 连接成功（${testResult.text?.model ?? testResult.model}）`
                  : `✗ ${testResult.text?.error ?? testResult.error}`}
              </div>
              {testResult.vision !== undefined && (
                testResult.vision === null ? (
                  <div className="px-4 py-2 text-sm rounded-lg bg-slate-50 text-slate-500">
                    🖼️ 视觉模型：未配置（将走平台默认）
                  </div>
                ) : (
                  <div className={`px-4 py-2 text-sm rounded-lg ${testResult.vision.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                    🖼️ 视觉模型：{testResult.vision.ok
                      ? `✓ 连接成功（${testResult.vision.model}）`
                      : `✗ ${testResult.vision.error}`}
                    {testResult.vision.warning && (
                      <div className="mt-1 text-xs text-amber-600">⚠ {testResult.vision.warning}</div>
                    )}
                  </div>
                )
              )}
            </div>
          )}

          {loading ? (
            <div className="text-center text-slate-400 py-12">加载中...</div>
          ) : (
            <>
            {/* 对话模型供应商区 */}
            <Card className="p-5 space-y-3">
              <div className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <span className="text-base">💬</span> 对话模型供应商
                <span className="text-xs font-normal text-slate-400">主回答模型</span>
              </div>
              <Field label="供应商 binding（选择后自动填默认 base_url）">
                <select value={draft.binding} onChange={(e) => onPickBinding(e.target.value)} className={inputCls}>
                  <option value="">— 选择 —</option>
                  {providers.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}{s.default_api_base ? `  (${s.default_api_base})` : ''}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`API Key${apiKeySet ? '（已设置，留空保留不修改）' : '（留空走平台默认）'}`}>
                <input
                  type="password"
                  value={draft.api_key}
                  onChange={(e) => setDraft({ ...draft, api_key: e.target.value })}
                  className={inputCls}
                  placeholder={apiKeySet ? '••••••（已设置，留空保留）' : 'sk-...'}
                />
              </Field>
              <Field label="Base URL（留空用供应商默认端点）">
                <input value={draft.base_url} onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} className={inputCls} placeholder="https://..." />
              </Field>
              <Field label="对话模型 text_model（必填）">
                <input value={draft.text_model} onChange={(e) => setDraft({ ...draft, text_model: e.target.value })} className={inputCls} placeholder="deepseek-chat / gpt-4o / qwen-plus" />
              </Field>
            </Card>

            {/* 视觉模型供应商区（独立，可异于对话供应商） */}
            <Card className="p-5 space-y-3">
              <div className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <span className="text-base">🖼️</span> 视觉模型供应商
                <span className="text-xs font-normal text-slate-400">看图用，可与对话供应商不同</span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                当上面的对话模型不支持看图（如 deepseek）时，由该视觉模型把图片转成文字描述再交给对话模型回答。
                可填与对话不同的供应商（如对话 deepseek、视觉 dashscope 的 qwen-vl）。全部留空则回退平台默认视觉模型。
              </p>
              <Field label="视觉供应商 binding（选择后自动填默认 base_url）">
                <select value={draft.vision_binding} onChange={(e) => onPickVisionBinding(e.target.value)} className={inputCls}>
                  <option value="">— 选择（留空走平台默认） —</option>
                  {providers.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}{s.default_api_base ? `  (${s.default_api_base})` : ''}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`视觉 API Key${visionApiKeySet ? '（已设置，留空保留不修改）' : '（留空走平台默认）'}`}>
                <input
                  type="password"
                  value={draft.vision_api_key}
                  onChange={(e) => setDraft({ ...draft, vision_api_key: e.target.value })}
                  className={inputCls}
                  placeholder={visionApiKeySet ? '••••••（已设置，留空保留）' : 'sk-...'}
                />
              </Field>
              <Field label="视觉 Base URL（留空用供应商默认端点）">
                <input value={draft.vision_base_url} onChange={(e) => setDraft({ ...draft, vision_base_url: e.target.value })} className={inputCls} placeholder="https://..." />
              </Field>
              <Field label="视觉模型 vision_model（看图用）">
                <input value={draft.vision_model} onChange={(e) => setDraft({ ...draft, vision_model: e.target.value })} className={inputCls} placeholder="qwen-vl-plus / qwen-vl-max / gpt-4o" />
              </Field>
            </Card>

            {/* 保存 / 测试 */}
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={handleTest}
                disabled={testing}
                className="px-3 py-1.5 text-sm text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 disabled:opacity-50"
              >
                {testing ? '测试中...' : '测试连接'}
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-4 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? '保存中...' : '保存'}
              </button>
            </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
