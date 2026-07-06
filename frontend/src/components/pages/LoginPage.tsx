import { useState } from 'react'
import { GraduationCap, PartyPopper } from 'lucide-react'
import { applyTeacher, login, register } from '../../services/auth'
import { Button, Input, Textarea } from '../ui'
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
      <div className="flex items-center justify-center min-h-screen bg-canvas px-4">
        <div className="w-full max-w-md px-5 md:px-8 py-10 bg-surface rounded-[var(--radius-lg)] border border-line text-center">
          <PartyPopper size={40} strokeWidth={1.5} className="text-ink mx-auto mb-4" />
          <h1 className="font-serif text-2xl text-ink mb-2">账号创建成功</h1>
          <p className="text-sm text-ink-soft mb-1">您的教师申请已提交。</p>
          <p className="text-xs text-muted mb-6">
            管理员审批通过后，您将获得教师权限，并收到站内通知。
          </p>
          <Button variant="primary" className="w-full" onClick={() => onLogin(registeredUser)}>
            进入学习
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-canvas px-4">
      <div className="w-full max-w-md px-5 md:px-8 py-8 md:py-10 bg-surface rounded-[var(--radius-lg)] border border-line">
        <div className="text-center mb-8">
          <GraduationCap size={36} strokeWidth={1.5} className="text-ink mx-auto mb-3" />
          <h1 className="font-serif text-2xl text-ink">课程学习 Agent</h1>
          <p className="text-sm text-muted mt-1">
            {mode === 'login' ? '登录以开始学习' : '创建一个新账号'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="用户名"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="请输入用户名"
            autoFocus
          />

          {mode === 'register' && (
            <>
              <Input
                label="显示名称（可选）"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="显示在界面上的名字"
              />

              {/* 主入口：申请教师权限（申请-审批流） */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={wantsTeacher}
                  onChange={(e) => setWantsTeacher(e.target.checked)}
                  className="rounded border-line accent-[var(--color-accent)]"
                />
                <span className="text-sm text-ink-soft">我是一名教师，申请教师权限</span>
              </label>
              {wantsTeacher && (
                <Textarea
                  label="申请理由"
                  value={teacherReason}
                  onChange={(e) => setTeacherReason(e.target.value)}
                  rows={3}
                  placeholder="请简要说明您的任教课程 / 院系，便于管理员审批"
                />
              )}

              {/* 次要入口：邀请码快速通道（折叠，与申请-审批并存） */}
              <button
                type="button"
                onClick={() => setShowInviteCode(!showInviteCode)}
                className="text-xs text-muted hover:text-ink-soft transition"
              >
                {showInviteCode ? '▼' : '▶'} 我有教师邀请码（快速通道，免审批）
              </button>
              {showInviteCode && (
                <Input
                  label="教师邀请码"
                  type="text"
                  value={inviteCode}
                  onChange={(e) => setInviteCode(e.target.value.toUpperCase())}
                  className="font-mono tracking-wider"
                  placeholder="填写后直接注册为教师"
                  maxLength={16}
                />
              )}
            </>
          )}

          <Input
            label="密码"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={mode === 'register' ? '至少 4 个字符' : '请输入密码'}
          />

          {error && (
            <div className="text-sm text-danger-fg bg-danger-bg rounded-[var(--radius)] px-4 py-2">
              {error}
            </div>
          )}

          <Button type="submit" variant="primary" loading={loading} className="w-full">
            {loading ? '处理中...' : mode === 'login' ? '登 录' : '注 册'}
          </Button>
        </form>

        <div className="text-center mt-6">
          <button
            type="button"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login')
              setError('')
            }}
            className="text-sm text-ink-soft hover:text-ink transition"
          >
            {mode === 'login' ? '没有账号？点击注册' : '已有账号？点击登录'}
          </button>
        </div>
      </div>
    </div>
  )
}
