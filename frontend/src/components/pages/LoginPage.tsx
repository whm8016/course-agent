import { useState } from 'react'
import { applyTeacher, login, register } from '../../services/auth'
import type { User } from '../../types'

interface Props {
  onLogin: (user: User) => void
}

export default function LoginPage({ onLogin }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [inviteCode, setInviteCode] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // 注册页两通道：申请教师（主入口）+ 邀请码快速通道（次要，折叠）
  const [wantsTeacher, setWantsTeacher] = useState(false)
  const [teacherReason, setTeacherReason] = useState('')
  const [showInviteCode, setShowInviteCode] = useState(false)
  // 注册 + 申请成功后的确认态：不立即进入，让用户明确知道申请已提交
  const [applied, setApplied] = useState(false)
  const [registeredUser, setRegisteredUser] = useState<User | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password.trim()) {
      setError('请输入用户名和密码')
      return
    }
    setError('')
    setLoading(true)
    try {
      if (mode === 'login') {
        const data = await login(username.trim(), password)
        onLogin(data.user)
      } else {
        if (password.length < 4) {
          setError('密码至少 4 个字符')
          setLoading(false)
          return
        }
        const data = await register(
          username.trim(),
          password,
          displayName.trim(),
          inviteCode.trim() || undefined,
        )
        // 邀请码快速通道已让用户成为 teacher，无需再申请
        // 理由留空不再静默跳过：用占位文案兜底提交（后端 reason 字段要求非空）
        if (data.user.role !== 'teacher' && wantsTeacher) {
          try {
            await applyTeacher(teacherReason.trim() || '（未填写申请理由）')
            setRegisteredUser(data.user)
            setApplied(true)
            return
          } catch (e2: unknown) {
            // 注册已成功，仅申请失败：仍进入，但提示用户
            setError(
              `账号已创建，但教师申请提交失败：${e2 instanceof Error ? e2.message : '请稍后重试'}`,
            )
            onLogin(data.user)
            return
          }
        }
        onLogin(data.user)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '操作失败')
    } finally {
      setLoading(false)
    }
  }

  // 注册 + 教师申请成功后的确认页（不直接跳转，让用户确认申请状态）
  if (applied && registeredUser) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 via-white to-slate-100">
        <div className="w-full max-w-md px-5 md:px-8 py-10 mx-4 md:mx-0 bg-white rounded-2xl shadow-xl border border-slate-100 text-center">
          <div className="text-5xl mb-4">🎉</div>
          <h1 className="text-2xl font-bold text-slate-800 mb-2">账号创建成功</h1>
          <p className="text-sm text-slate-600 mb-1">您的教师申请已提交。</p>
          <p className="text-xs text-slate-400 mb-6">
            管理员审批通过后，您将获得教师权限，并收到站内通知。
          </p>
          <button
            onClick={() => onLogin(registeredUser)}
            className="w-full py-2.5 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition"
          >
            进入学习
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 via-white to-slate-100">
      <div className="w-full max-w-md px-5 md:px-8 py-8 md:py-10 mx-4 md:mx-0 bg-white rounded-2xl shadow-xl border border-slate-100">
        <div className="text-center mb-8">
          <div className="text-4xl mb-3">📚</div>
          <h1 className="text-2xl font-bold text-slate-800">课程学习 Agent</h1>
          <p className="text-sm text-slate-400 mt-1">
            {mode === 'login' ? '登录以开始学习' : '创建一个新账号'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">用户名</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
              placeholder="请输入用户名"
              autoFocus
            />
          </div>

          {mode === 'register' && (
            <>
              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">显示名称（可选）</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
                  placeholder="显示在界面上的名字"
                />
              </div>

              {/* 主入口：申请教师权限（申请-审批流） */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={wantsTeacher}
                  onChange={(e) => setWantsTeacher(e.target.checked)}
                  className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-100"
                />
                <span className="text-sm text-slate-600">我是一名教师，申请教师权限</span>
              </label>
              {wantsTeacher && (
                <div>
                  <label className="block text-sm font-medium text-slate-600 mb-1">申请理由</label>
                  <textarea
                    value={teacherReason}
                    onChange={(e) => setTeacherReason(e.target.value)}
                    rows={3}
                    className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition resize-none"
                    placeholder="请简要说明您的任教课程 / 院系，便于管理员审批"
                  />
                </div>
              )}

              {/* 次要入口：邀请码快速通道（折叠，与申请-审批并存） */}
              <button
                type="button"
                onClick={() => setShowInviteCode(!showInviteCode)}
                className="text-xs text-slate-400 hover:text-slate-600 transition"
              >
                {showInviteCode ? '▼' : '▶'} 我有教师邀请码（快速通道，免审批）
              </button>
              {showInviteCode && (
                <div>
                  <label className="block text-sm font-medium text-slate-600 mb-1">
                    教师邀请码
                  </label>
                  <input
                    type="text"
                    value={inviteCode}
                    onChange={(e) => setInviteCode(e.target.value.toUpperCase())}
                    className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition font-mono tracking-wider"
                    placeholder="填写后直接注册为教师"
                    maxLength={16}
                  />
                </div>
              )}
            </>
          )}

          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">密码</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
              placeholder={mode === 'register' ? '至少 4 个字符' : '请输入密码'}
            />
          </div>

          {error && (
            <div className="text-sm text-red-500 bg-red-50 rounded-lg px-4 py-2">{error}</div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? '处理中...' : mode === 'login' ? '登 录' : '注 册'}
          </button>
        </form>

        <div className="text-center mt-6">
          <button
            type="button"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login')
              setError('')
            }}
            className="text-sm text-indigo-600 hover:text-indigo-800 transition"
          >
            {mode === 'login' ? '没有账号？点击注册' : '已有账号？点击登录'}
          </button>
        </div>
      </div>
    </div>
  )
}
