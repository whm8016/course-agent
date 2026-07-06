import { useState } from 'react'
import { GraduationCap, Briefcase, Settings, X, Plus } from 'lucide-react'
import type { Course, Session, User } from '../../types'
import CourseSelector from './CourseSelector'
import SessionList from './SessionList'
import { authHeaders } from '../../services/auth'
import { Avatar, Badge, Button } from '../ui'

interface Props {
  courses: Course[]
  activeCourseId: string
  onSelectCourse: (id: string) => void
  sessions: Session[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onCreateSession: () => void
  onDeleteSession: (id: string) => void
  user: User
  onLogout: () => void
  onCoursesRefresh?: () => void
  /** 打开「工作台」聚合页（仪表盘/图谱/知识包/笔记本/教师/管理）*/
  onWorkbench: () => void
  /** 打开「设置」聚合页（模型/MCP/搜索/Bot/通知）*/
  onSettings: () => void
  notifUnread?: number
  onCloseMobile?: () => void
}

export default function Sidebar({
  courses,
  activeCourseId,
  onSelectCourse,
  sessions,
  activeSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  user,
  onLogout,
  onCoursesRefresh,
  onWorkbench,
  onSettings,
  notifUnread,
  onCloseMobile,
}: Props) {
  const [joinCode, setJoinCode] = useState('')
  const [joinMsg, setJoinMsg] = useState('')
  const [joining, setJoining] = useState(false)
  const [showJoin, setShowJoin] = useState(false)

  const role = user.role ?? (user.is_admin ? 'admin' : 'student')
  const isTeacher = role === 'teacher' || role === 'admin'

  const handleJoin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!joinCode.trim()) return
    setJoining(true)
    setJoinMsg('')
    try {
      const res = await fetch('/api/courses/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ join_code: joinCode.trim() }),
      })
      const data = await res.json() as { name?: string; already_enrolled?: boolean; detail?: string }
      if (!res.ok) throw new Error(data.detail || '加入失败')
      setJoinMsg(data.already_enrolled ? `已在「${data.name}」` : `已加入「${data.name}」`)
      setJoinCode('')
      onCoursesRefresh?.()
    } catch (err) {
      setJoinMsg(err instanceof Error ? err.message : '加入失败')
    } finally {
      setJoining(false)
    }
  }

  const handleSelectCourse = (id: string) => {
    onSelectCourse(id)
    onCloseMobile?.()
  }

  const handleSelectSession = (id: string) => {
    onSelectSession(id)
    onCloseMobile?.()
  }

  return (
    <aside className="w-64 h-full bg-surface border-r border-line flex flex-col overflow-hidden">
      <div className="px-4 py-5 border-b border-line">
        <div className="flex items-center justify-between">
          <h1 className="font-serif text-base text-ink flex items-center gap-2">
            <GraduationCap size={20} strokeWidth={1.5} className="text-ink" />
            课程学习 Agent
          </h1>
          <button
            onClick={onCloseMobile}
            className="md:hidden p-1 -mr-1 rounded-[var(--radius-sm)] text-muted hover:text-ink hover:bg-surface-2 transition"
            aria-label="关闭侧边栏"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <p className="text-xs text-muted mt-1">多 Agent 编排 · RAG 检索</p>
      </div>

      <div className="max-h-[35vh] overflow-y-auto p-3 border-b border-line">
        <CourseSelector
          courses={[
            { id: 'general', name: '自由问答', icon: '💬', description: '通用学习问答（未选课）', source: 'builtin' },
            ...courses.filter((c) => c.id !== 'general'),
          ]}
          activeCourseId={activeCourseId}
          onSelect={handleSelectCourse}
        />
      </div>

      {/* 学生用课程码加入 */}
      {!isTeacher && (
        <div className="px-3 py-2 border-b border-line">
          {showJoin ? (
            <form onSubmit={handleJoin} className="space-y-1.5">
              <input
                value={joinCode}
                onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                className="w-full bg-surface border border-line rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs font-mono text-ink placeholder:text-muted focus:outline-none focus:border-ink transition"
                placeholder="课程码 XXXXXXXX"
                maxLength={16}
                autoFocus
              />
              <div className="flex gap-1.5">
                <Button type="submit" variant="primary" size="sm" loading={joining} className="flex-1">
                  {joining ? '加入中...' : '加入'}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowJoin(false)
                    setJoinMsg('')
                  }}
                >
                  取消
                </Button>
              </div>
              {joinMsg && <p className="text-xs text-ink-soft">{joinMsg}</p>}
            </form>
          ) : (
            <button
              onClick={() => setShowJoin(true)}
              className="w-full text-xs text-ink-soft hover:text-ink py-1.5 rounded-[var(--radius-sm)] hover:bg-surface-2 transition text-left px-1 flex items-center gap-1"
            >
              <Plus size={12} strokeWidth={1.5} />
              用课程码加入课程
            </button>
          )}
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-3">
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelect={handleSelectSession}
          onCreate={onCreateSession}
          onDelete={onDeleteSession}
        />
      </div>

      <div className="p-4 border-t border-line">
        <div className="flex items-center gap-2.5 mb-3">
          <Avatar name={user.display_name || user.username} size={32} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5">
              <span
                className="text-sm text-ink font-medium truncate"
                title={user.display_name || user.username}
              >
                {user.display_name || user.username}
              </span>
              {role === 'admin' && <Badge color="neutral">管理员</Badge>}
              {role === 'teacher' && <Badge color="info">教师</Badge>}
            </div>
          </div>
          <button
            onClick={onLogout}
            className="text-xs text-muted hover:text-danger-fg transition shrink-0"
          >
            退出
          </button>
        </div>

        {/* 双聚合入口：工作台 / 设置。其余十项功能全部收进这两个 overlay。 */}
        <div className="grid grid-cols-2 gap-1.5">
          <Button variant="ghost" size="sm" icon={Briefcase} onClick={onWorkbench} className="justify-center">
            工作台
          </Button>
          <Button
            variant="ghost"
            size="sm"
            icon={Settings}
            onClick={onSettings}
            className="relative justify-center"
          >
            设置
            {(notifUnread ?? 0) > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-danger-fg ring-2 ring-surface" />
            )}
          </Button>
        </div>
      </div>
    </aside>
  )
}
