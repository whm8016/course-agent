import { useEffect, useState, useCallback, useRef } from 'react'
import {
  fetchBots, createBot, startBot, stopBot, deleteBot, sendBotMessage, updateBot, fetchBotHistory,
  listReminders, createReminder, deleteReminder,
  generateBindCode, fetchMyBindings, deleteBinding,
  type BotInstance, type BotReminder, type BotReminderSchedule, type SocialBinding,
} from '../../services/api'
import { Modal, Badge, Card, EmptyState, StatusDot } from '../ui'
import FormattedMarkdown from '../shared/FormattedMarkdown'

interface Props {
  onBack: () => void
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChannelField {
  k: string
  label: string
  placeholder?: string
  secret?: boolean
}

interface ChannelDef {
  key: string
  label: string
  hint: string
  fields: ChannelField[]
}

const CHANNELS: ChannelDef[] = [
  { key: 'web', label: 'Web', hint: '网页内对话，无需凭证', fields: [] },
  {
    key: 'qq',
    label: 'QQ',
    hint: '在 QQ 开放平台创建机器人后，填入你自己的凭证',
    fields: [
      { k: 'app_id', label: 'App ID', placeholder: 'QQ 机器人 AppID' },
      { k: 'secret', label: 'App Secret', placeholder: 'QQ 机器人 Token / Secret', secret: true },
    ],
  },
  {
    key: 'feishu',
    label: '飞书',
    hint: '在飞书开放平台创建应用后，填入你自己的凭证',
    fields: [
      { k: 'app_id', label: 'App ID', placeholder: 'cli_xxxxxx' },
      { k: 'app_secret', label: 'App Secret', placeholder: '飞书 App Secret', secret: true },
    ],
  },
]

export default function BotPage({ onBack }: Props) {
  const [bots, setBots] = useState<BotInstance[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [creating, setCreating] = useState(false)
  const [chattingBot, setChattingBot] = useState<BotInstance | null>(null)
  const [reminderBot, setReminderBot] = useState<BotInstance | null>(null)
  const [editingBot, setEditingBot] = useState<BotInstance | null>(null)
  const [showBinding, setShowBinding] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setBots(await fetchBots())
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const handleToggle = async (b: BotInstance) => {
    try {
      if (b.running) {
        await stopBot(b.bot_id)
      } else {
        await startBot(b.bot_id)
      }
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '操作失败')
    }
  }

  const handleDelete = async (b: BotInstance) => {
    if (!confirm(`确认删除 Bot「${b.name}」？\n此操作会停止它并清除持久化配置和会话记录，不可恢复。`)) return
    try {
      await deleteBot(b.bot_id)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="text-slate-400 hover:text-slate-600 text-sm">
            ← 返回
          </button>
          <h1 className="text-lg font-semibold text-slate-800">🤖 我的 Bot</h1>
          <Badge color="indigo">{bots.length}</Badge>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowBinding(true)}
            className="px-3 py-1.5 text-sm bg-white border border-slate-200 text-slate-600 rounded-lg hover:border-indigo-300 hover:text-indigo-600 transition"
          >
            🔗 绑定 IM
          </button>
          <button
            onClick={() => setCreating(true)}
            className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
          >
            + 新建 Bot
          </button>
        </div>
      </header>
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mb-4 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-800 space-y-1">
          <p className="font-medium">💡 Bot = 你的专属对话机器人（共享后端 agent 引擎：安全护栏/课程提示/记忆）</p>
          <p>• 每人独立管理自己的 Bot，互不可见。点「💬 对话」即可在网页内与它交流。</p>
          <p>• <b>QQ / 飞书</b>：新建 Bot 时填入你自己机器人的 App ID / Secret（每个 Bot 独立持有，存于本机配置，不再用全局 .env）。</p>
          <p>• <b>Web</b>：网页内对话，无需额外凭证。</p>
          <p>• <b>会话隔离</b>：网页「💬 对话」与 QQ / 飞书各频道会话<b>互相独立</b>（QQ 群聊历史不在网页显示，反之亦然）——网页窗口用于测试 bot / 独立 Web 对话，每个频道、每个群各自独立会话是标准设计。</p>
        </div>
        {error && (
          <div className="mb-4 px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>
        )}
        {loading ? (
          <div className="text-center text-slate-400 py-16">加载中...</div>
        ) : bots.length === 0 ? (
          <EmptyState icon="🤖" title="还没有 Bot" hint="点击右上角新建" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {bots.map((b) => {
              const chans = Array.isArray(b.channels) ? b.channels : Object.keys(b.channels || {})
              return (
                <Card key={b.bot_id} className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <StatusDot status={b.running ? 'connected' : 'disabled'} />
                    <h3 className="font-semibold text-slate-800 truncate">{b.name}</h3>
                    {b.running ? (
                      <Badge color="green">运行中</Badge>
                    ) : (
                      <Badge color="slate">已停止</Badge>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 font-mono mb-1 truncate">{b.bot_id}</p>
                  <p className="text-xs text-slate-500 line-clamp-2 mb-2">
                    {b.description || '（无描述）'}
                  </p>
                  <div className="flex flex-wrap gap-1 mb-3">
                    {chans.length > 0 ? (
                      chans.map((ch) => (
                        <Badge key={ch} color="blue">
                          {ch}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-xs text-slate-400">无频道</span>
                    )}
                    {b.course_id && <Badge color="indigo">课程绑定</Badge>}
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setChattingBot(b)}
                        className="flex-1 text-xs py-1.5 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 rounded transition"
                      >
                        💬 对话
                      </button>
                      <button
                        onClick={() => setReminderBot(b)}
                        className="flex-1 text-xs py-1.5 bg-amber-50 text-amber-700 hover:bg-amber-100 rounded transition"
                      >
                        ⏰ 提醒
                      </button>
                      <button
                        onClick={() => setEditingBot(b)}
                        className="flex-1 text-xs py-1.5 bg-slate-100 text-slate-600 hover:bg-slate-200 rounded transition"
                      >
                        ✏️ 编辑
                      </button>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleToggle(b)}
                        className={`flex-1 text-xs py-1.5 rounded transition ${
                          b.running
                            ? 'bg-red-50 text-red-600 hover:bg-red-100'
                            : 'bg-green-50 text-green-700 hover:bg-green-100'
                        }`}
                      >
                        {b.running ? '停止' : '启动'}
                      </button>
                      <button
                        onClick={() => handleDelete(b)}
                        className="text-xs px-3 py-1.5 bg-slate-50 text-slate-500 hover:bg-red-50 hover:text-red-600 rounded transition"
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
      {(creating || editingBot) && (
        <BotEditor
          existingBot={editingBot ?? undefined}
          onClose={() => {
            setCreating(false)
            setEditingBot(null)
          }}
          onSaved={() => {
            setCreating(false)
            setEditingBot(null)
            void reload()
          }}
        />
      )}
      {chattingBot && (
        <BotChat bot={chattingBot} onClose={() => setChattingBot(null)} />
      )}
      {reminderBot && (
        <BotReminderPanel bot={reminderBot} onClose={() => setReminderBot(null)} />
      )}
      {showBinding && (
        <BotBindingPanel onClose={() => setShowBinding(false)} />
      )}
    </div>
  )
}

function BotChat({ bot, onClose }: { bot: BotInstance; onClose: () => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 打开时加载历史（解决关闭重开记录丢失；后端 per-user session 已持久化）
  useEffect(() => {
    let cancelled = false
    fetchBotHistory(bot.bot_id)
      .then((rows) => {
        if (cancelled || rows.length === 0) return
        setMessages(rows.map((r) => ({ role: r.role as 'user' | 'assistant', content: r.content })))
      })
      .catch(() => { /* 静默 */ })
    return () => { cancelled = true }
  }, [bot.bot_id])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, sending])

  const send = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setSending(true)
    try {
      const reply = await sendBotMessage(bot.bot_id, text)
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `⚠️ ${e instanceof Error ? e.message : '发送失败'}` },
      ])
    } finally {
      setSending(false)
    }
  }

  return (
    <Modal
      open
      onClose={onClose}
      overlayCloses={false}
      title={`💬 ${bot.name}`}
      footer={
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700"
        >
          关闭
        </button>
      }
    >
      <div className="flex flex-col" style={{ height: '60vh' }}>
        <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-3 pr-1">
          {messages.length === 0 ? (
            <p className="text-center text-slate-400 text-sm py-12">
              和「{bot.name}」开始对话吧 ✨
            </p>
          ) : (
            messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] px-3 py-2 rounded-2xl text-sm ${
                    m.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-100 text-slate-800'
                  }`}
                >
                  {m.role === 'assistant' ? (
                    <FormattedMarkdown content={m.content} />
                  ) : (
                    <span className="whitespace-pre-wrap">{m.content}</span>
                  )}
                </div>
              </div>
            ))
          )}
          {sending && (
            <div className="text-center text-xs text-slate-400 animate-pulse">思考中...</div>
          )}
        </div>
        <div className="mt-3 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
            disabled={sending}
            className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 disabled:bg-slate-50"
            placeholder="输入消息，Enter 发送"
          />
          <button
            onClick={send}
            disabled={sending || !input.trim()}
            className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            发送
          </button>
        </div>
      </div>
    </Modal>
  )
}

function formatReminderSchedule(schedule: BotReminderSchedule, state: BotReminder['state']): string {
  const parts: string[] = []
  if (schedule.kind === 'every' && schedule.every_seconds) {
    parts.push(`每 ${Math.round(schedule.every_seconds / 60)} 分钟`)
  } else if (schedule.kind === 'cron' && schedule.expr) {
    parts.push(`cron ${schedule.expr}`)
  } else if (schedule.kind === 'at') {
    parts.push('一次性')
  } else {
    parts.push(schedule.kind || '?')
  }
  if (state.next_run_at_ms) {
    const d = new Date(state.next_run_at_ms)
    parts.push(
      `下次 ${d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}`
    )
  }
  return parts.join(' · ')
}

function BotReminderPanel({ bot, onClose }: { bot: BotInstance; onClose: () => void }) {
  const [reminders, setReminders] = useState<BotReminder[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [kind, setKind] = useState<'repeat' | 'once' | 'cron'>('repeat')
  const [delayMin, setDelayMin] = useState('5')
  const [everyMin, setEveryMin] = useState('30')
  const [cronExpr, setCronExpr] = useState('0 9 * * *')
  const [channel, setChannel] = useState<'web' | 'qq' | 'feishu'>('web')
  const [chatId, setChatId] = useState('')
  const [saving, setSaving] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setReminders(await listReminders(bot.bot_id))
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [bot.bot_id])

  useEffect(() => {
    void reload()
  }, [reload])

  const handleCreate = async () => {
    if (!message.trim()) {
      setError('提醒内容不能为空')
      return
    }
    const schedule: BotReminderSchedule =
      kind === 'once'
        ? { kind: 'at', at_ms: Date.now() + Number(delayMin) * 60000, every_seconds: null, expr: null, tz: null }
        : kind === 'repeat'
        ? { kind: 'every', at_ms: null, every_seconds: Number(everyMin) * 60, expr: null, tz: null }
        : { kind: 'cron', at_ms: null, every_seconds: null, expr: cronExpr.trim(), tz: 'Asia/Shanghai' }
    setSaving(true)
    setError('')
    try {
      if (channel !== 'web' && !chatId.trim()) {
        setError('IM 渠道需填写 chat_id（QQ 群 openid / 用户 openid）')
        return
      }
      await createReminder(bot.bot_id, { message: message.trim(), channel, chat_id: chatId.trim(), schedule })
      setMessage('')
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '创建失败')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteReminder(bot.bot_id, id)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败')
    }
  }

  return (
    <Modal
      open
      onClose={onClose}
      overlayCloses={false}
      title={`⏰ ${bot.name} 的提醒`}
      footer={
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700"
        >
          关闭
        </button>
      }
    >
      <div className="space-y-3">
        {error && <div className="px-3 py-2 bg-red-50 text-red-600 text-xs rounded">{error}</div>}
        <div className="p-3 bg-slate-50 rounded-lg space-y-2">
          <div className="flex gap-1.5 text-xs">
            {(['web', 'qq', 'feishu'] as const).map((ch) => (
              <button
                key={ch}
                onClick={() => setChannel(ch)}
                className={`px-2 py-1 rounded transition ${
                  channel === ch ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-600'
                }`}
              >
                {ch === 'web' ? '网页' : ch === 'qq' ? 'QQ' : '飞书'}
              </button>
            ))}
          </div>
          {channel !== 'web' && (
            <input
              value={chatId}
              onChange={(e) => setChatId(e.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-xs font-mono focus:outline-none focus:border-indigo-400"
              placeholder={channel === 'qq' ? 'QQ 群 openid / 用户 openid' : '飞书 chat_id / open_id'}
            />
          )}
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={2}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
            placeholder="提醒内容，如：该复习电路第二章了"
          />
          <div className="flex gap-1.5 text-xs">
            {(['repeat', 'once', 'cron'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setKind(k)}
                className={`px-2 py-1 rounded transition ${
                  kind === k ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-600'
                }`}
              >
                {k === 'repeat' ? '周期重复' : k === 'once' ? '一次性' : '定时 Cron'}
              </button>
            ))}
          </div>
          {kind === 'once' && (
            <div className="space-y-1">
              <input
                value={delayMin}
                onChange={(e) => setDelayMin(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
                placeholder="输入分钟数，如 5 表示 5 分钟后提醒"
              />
              <p className="text-[11px] text-slate-400">⏱ 多少分钟后触发一次提醒</p>
            </div>
          )}
          {kind === 'repeat' && (
            <div className="space-y-1">
              <input
                value={everyMin}
                onChange={(e) => setEveryMin(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
                placeholder="输入分钟数，如 30 表示每 30 分钟提醒一次"
              />
              <p className="text-[11px] text-slate-400">🔄 每隔多少分钟重复提醒</p>
            </div>
          )}
          {kind === 'cron' && (
            <div className="space-y-1">
              <input
                value={cronExpr}
                onChange={(e) => setCronExpr(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
                placeholder="分 时 日 月 周"
              />
              <p className="text-[11px] text-slate-400">
                📅 Cron 表达式，如 <code className="bg-slate-100 px-1 rounded">0 9 * * *</code> = 每天上午 9:00
              </p>
              <div className="text-[10px] text-slate-400 bg-slate-50 rounded p-2 space-y-0.5">
                <p>示例：</p>
                <p>• <code>0 9 * * *</code> → 每天 9:00</p>
                <p>• <code>30 14 * * 1-5</code> → 周一到周五 14:30</p>
                <p>• <code>0 */2 * * *</code> → 每 2 小时</p>
              </div>
            </div>
          )}
          <button
            onClick={handleCreate}
            disabled={saving}
            className="w-full py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            {saving ? '添加中...' : '+ 添加提醒'}
          </button>
        </div>
        {loading ? (
          <div className="text-center text-slate-400 text-sm py-4">加载中...</div>
        ) : reminders.length === 0 ? (
          <p className="text-center text-slate-400 text-sm py-4">还没有提醒</p>
        ) : (
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {reminders.map((r) => (
              <div key={r.id} className="p-2 border border-slate-200 rounded-lg flex justify-between items-start gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-700">{r.message}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{formatReminderSchedule(r.schedule, r.state)}</p>
                </div>
                <button
                  onClick={() => handleDelete(r.id)}
                  className="text-xs text-red-500 hover:bg-red-50 px-2 py-1 rounded shrink-0"
                >
                  删除
                </button>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-slate-400">
          web 渠道的提醒到点后会出现在侧边栏通知铃铛里；QQ/飞书渠道需绑定账号。
        </p>
      </div>
    </Modal>
  )
}

function BotBindingPanel({ onClose }: { onClose: () => void }) {
  const [code, setCode] = useState('')
  const [expires, setExpires] = useState(0)
  const [bindings, setBindings] = useState<SocialBinding[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [genLoading, setGenLoading] = useState(false)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      setBindings(await fetchMyBindings())
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

  const handleGen = async () => {
    setGenLoading(true)
    setError('')
    try {
      const r = await generateBindCode()
      setCode(r.code)
      setExpires(r.expires_in || 600)
    } catch (e) {
      setError(e instanceof Error ? e.message : '生成失败')
    } finally {
      setGenLoading(false)
    }
  }

  const handleUnbind = async (id: string) => {
    if (!confirm('确认解绑该 IM 账号？解绑后该渠道将不再共享记忆。')) return
    try {
      await deleteBinding(id)
      void reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : '解绑失败')
    }
  }

  return (
    <Modal
      open
      onClose={onClose}
      overlayCloses={false}
      title="🔗 绑定 IM 账号"
      footer={
        <button onClick={onClose} className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700">
          关闭
        </button>
      }
    >
      <div className="space-y-3">
        <p className="text-xs text-slate-500">
          把 QQ / 飞书账号绑定到网站账号后，bot 会<b>跨渠道记住你</b>（学习画像、知识图谱长期共享）。
          短期对话历史仍各渠道独立（对齐 DeepTutor session≠memory 分层）。
        </p>
        {error && <div className="px-3 py-2 bg-red-50 text-red-600 text-xs rounded">{error}</div>}
        <div className="p-3 bg-slate-50 rounded-lg space-y-2">
          <button
            onClick={handleGen}
            disabled={genLoading}
            className="w-full py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            {genLoading ? '生成中...' : '生成绑定码'}
          </button>
          {code && (
            <div className="text-center space-y-1">
              <p className="text-xs text-slate-500">在 QQ / 飞书私聊你的 bot 发送：</p>
              <p className="text-2xl font-mono font-bold tracking-widest text-indigo-600">绑定 {code}</p>
              <p className="text-[11px] text-slate-400">{Math.round(expires / 60)} 分钟内有效，一次性</p>
            </div>
          )}
        </div>
        <div>
          <p className="text-xs font-medium text-slate-600 mb-1">已绑定账号</p>
          {loading ? (
            <p className="text-center text-slate-400 text-sm py-3">加载中...</p>
          ) : bindings.length === 0 ? (
            <p className="text-center text-slate-400 text-sm py-3">尚未绑定任何 IM 账号</p>
          ) : (
            <div className="space-y-1.5">
              {bindings.map((b) => (
                <div key={b.id} className="p-2 border border-slate-200 rounded-lg flex justify-between items-center gap-2">
                  <div className="flex-1 min-w-0">
                    <span className="text-sm text-slate-700">{b.platform}</span>
                    <span className="text-xs text-slate-400 font-mono ml-2 truncate">{b.platform_user_id}</span>
                  </div>
                  <button
                    onClick={() => handleUnbind(b.id)}
                    className="text-xs text-red-500 hover:bg-red-50 px-2 py-1 rounded shrink-0"
                  >
                    解绑
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

function BotEditor({
  onClose,
  onSaved,
  existingBot,
}: {
  onClose: () => void
  onSaved: () => void
  existingBot?: BotInstance
}) {
  const editing = Boolean(existingBot)
  const [botId, setBotId] = useState(existingBot?.bot_id ?? '')
  const [name, setName] = useState(existingBot?.name ?? '')
  const [description, setDescription] = useState(existingBot?.description ?? '')
  // persona 接口不回显（安全）；编辑时留空 = 保持不变
  const [persona, setPersona] = useState('')
  const [courseId, setCourseId] = useState(existingBot?.course_id ?? '')
  // channels 仅新建时可配（编辑不改，避免凭证丢失；要换频道请删除重建）
  const [channels, setChannels] = useState<Record<string, Record<string, string | boolean>>>({
    web: { enabled: true },
  })
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')

  const toggleChannel = (ch: ChannelDef) => {
    setChannels((prev) => {
      const next = { ...prev }
      if (next[ch.key]?.enabled) {
        delete next[ch.key]
      } else {
        const init: Record<string, string | boolean> = { enabled: true }
        for (const f of ch.fields) init[f.k] = ''
        next[ch.key] = init
      }
      return next
    })
  }

  const setChannelField = (key: string, fieldK: string, value: string) => {
    setChannels((prev) => ({ ...prev, [key]: { ...prev[key], [fieldK]: value } }))
  }

  const submit = async () => {
    if (!botId.trim() || !name.trim()) {
      setErr('bot_id 和 name 不能为空')
      return
    }
    setSaving(true)
    setErr('')
    try {
      if (editing) {
        const payload: {
          name: string
          description: string
          course_id: string
          persona?: string
        } = { name: name.trim(), description, course_id: courseId }
        if (persona.trim()) payload.persona = persona.trim()
        await updateBot(existingBot!.bot_id, payload)
      } else {
        // 新建：校验 IM 频道凭证
        for (const ch of CHANNELS) {
          if (channels[ch.key]?.enabled) {
            for (const f of ch.fields) {
              if (!String(channels[ch.key]?.[f.k] ?? '').trim()) {
                setErr(`${ch.label} 频道需填写 ${f.label}`)
                setSaving(false)
                return
              }
            }
          }
        }
        await createBot({
          bot_id: botId.trim(),
          name: name.trim(),
          description,
          persona,
          course_id: courseId,
          channels,
        })
      }
      onSaved()
    } catch (e) {
      setErr(e instanceof Error ? e.message : editing ? '保存失败' : '创建失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      open
      onClose={onClose}
      overlayCloses={false}
      title={editing ? `编辑 ${existingBot?.name}` : '新建 Bot'}
      footer={
        <>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-700"
          >
            取消
          </button>
          <button
            onClick={submit}
            disabled={saving}
            className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
          >
            {saving ? (editing ? '保存中...' : '创建中...') : editing ? '保存' : '创建'}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        {err && <div className="px-3 py-2 bg-red-50 text-red-600 text-xs rounded">{err}</div>}
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Bot ID</label>
          <input
            value={botId}
            onChange={(e) => setBotId(e.target.value)}
            disabled={editing}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400 disabled:bg-slate-50 disabled:text-slate-400"
            placeholder="如 study-buddy"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">名称</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
            placeholder="如 学习伙伴"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">描述</label>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
          />
        </div>
        {!editing && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              启用频道（可多选，建后不可改）
            </label>
            <div className="space-y-1.5">
              {CHANNELS.map((ch) => {
                const enabled = Boolean(channels[ch.key]?.enabled)
                return (
                  <div key={ch.key} className="p-2 rounded-lg border border-slate-200">
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={() => toggleChannel(ch)}
                        className="mt-0.5"
                      />
                      <div className="flex-1">
                        <div className="text-sm font-medium text-slate-700">{ch.label}</div>
                        <div className="text-xs text-slate-400">{ch.hint}</div>
                      </div>
                    </label>
                    {enabled && ch.fields.length > 0 && (
                      <div className="mt-2 ml-6 space-y-1.5">
                        {ch.fields.map((f) => (
                          <input
                            key={f.k}
                            type={f.secret ? 'password' : 'text'}
                            value={String(channels[ch.key]?.[f.k] ?? '')}
                            onChange={(e) => setChannelField(ch.key, f.k, e.target.value)}
                            className="w-full rounded border border-slate-200 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-indigo-400"
                            placeholder={f.placeholder}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
        {editing && (
          <p className="text-xs text-slate-400">
            频道配置建后不可改（避免凭证丢失）；如需换频道请删除重建。
          </p>
        )}
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">
            人设（persona{editing ? '，留空保持不变' : '，可选'}）
          </label>
          <textarea
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            rows={3}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400"
            placeholder={
              editing ? '如需修改人设请输入，留空保持不变' : '如：耐心、鼓励式的苏格拉底式导师'
            }
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">
            绑定课程 ID（可选，用于 RAG）
          </label>
          <input
            value={courseId}
            onChange={(e) => setCourseId(e.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-400"
          />
        </div>
      </div>
    </Modal>
  )
}
