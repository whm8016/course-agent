import { useState } from 'react'
import { login, register } from '../services/auth'
import type { User } from '../types'

interface Props {
  onLogin: (user: User) => void
}

export default function LoginPage({ onLogin }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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
        const data = await register(username.trim(), password, displayName.trim())
        onLogin(data.user)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '操作失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 via-white to-slate-100">
      <div className="w-full max-w-md px-8 py-10 bg-white rounded-2xl shadow-xl border border-slate-100">
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
