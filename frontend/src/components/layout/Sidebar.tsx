import { useState } from 'react'
import type { Course, Session, User } from '../../types'
import CourseSelector from './CourseSelector'
import SessionList from './SessionList'
import { authHeaders } from '../../services/auth'

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
  onAdmin?: () => void
  onTeacher?: () => void
  onCoursesRefresh?: () => void
  onDashboard?: () => void
  onGraph?: () => void
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
  onAdmin,
  onTeacher,
  onCoursesRefresh,
  onDashboard,
  onGraph,
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
    <aside className="w-64 h-full bg-white border-r border-slate-200 flex flex-col">
      <div className="px-4 py-5 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <h1 className="text-base font-bold text-slate-800 flex items-center gap-2">
            <span className="text-xl">📚</span>
            课程学习 Agent
          </h1>
          <button
            onClick={onCloseMobile}
            className="md:hidden p-1 text-slate-400 hover:text-slate-600 transition"
          >
            ✕
          </button>
        </div>
        <p className="text-xs text-slate-400 mt-1">LangGraph 多 Agent 编排</p>
      </div>

      <div className="p-3 border-b border-slate-100">
        <CourseSelector
          courses={courses}
          activeCourseId={activeCourseId}
          onSelect={handleSelectCourse}
        />
      </div>

      {/* 学生用课程码加入 */}
      {!isTeacher && (
        <div className="px-3 py-2 border-b border-slate-100">
          {showJoin ? (
            <form onSubmit={handleJoin} className="space-y-1.5">
              <input
                value={joinCode}
                onChange={e => setJoinCode(e.target.value.toUpperCase())}
                className="w-full rounded border border-slate-200 px-2.5 py-1.5 text-xs font-mono focus:outline-none focus:border-indigo-400 transition"
                placeholder="课程码 XXXXXXXX"
                maxLength={16}
                autoFocus
              />
              <div className="flex gap-1.5">
                <button
                  type="submit"
                  disabled={joining}
                  className="flex-1 text-xs py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition"
                >
                  {joining ? '加入中...' : '加入'}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowJoin(false); setJoinMsg('') }}
                  className="text-xs px-2 py-1.5 text-slate-400 hover:text-slate-600 transition"
                >
                  取消
                </button>
              </div>
              {joinMsg && <p className="text-xs text-indigo-600">{joinMsg}</p>}
            </form>
          ) : (
            <button
              onClick={() => setShowJoin(true)}
              className="w-full text-xs text-slate-500 hover:text-indigo-600 py-1.5 rounded hover:bg-indigo-50 transition text-left px-1"
            >
              + 用课程码加入课程
            </button>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3">
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelect={handleSelectSession}
          onCreate={onCreateSession}
          onDelete={onDeleteSession}
        />
      </div>

      <div className="p-4 border-t border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-600 font-medium truncate" title={user.display_name || user.username}>
            {user.display_name || user.username}
            {role === 'admin' && <span className="ml-1 text-xs text-purple-500">管理员</span>}
            {role === 'teacher' && <span className="ml-1 text-xs text-green-600">教师</span>}
          </span>
          <button
            onClick={onLogout}
            className="text-xs text-slate-400 hover:text-red-500 transition"
          >
            退出
          </button>
        </div>
        {(role === 'admin' || role === 'teacher') && onTeacher && (
          <button
            onClick={onTeacher}
            className="w-full text-xs text-center text-green-700 hover:text-green-900 py-1 rounded hover:bg-green-50 transition mb-1"
          >
            教师工作台
          </button>
        )}
        {role === 'admin' && onAdmin && (
          <button
            onClick={onAdmin}
            className="w-full text-xs text-center text-indigo-600 hover:text-indigo-800 py-1 rounded hover:bg-indigo-50 transition mb-1"
          >
            管理后台
          </button>
        )}
        {onDashboard && (
          <button
            onClick={onDashboard}
            className="w-full text-xs text-center text-teal-600 hover:text-teal-800 py-1 rounded hover:bg-teal-50 transition mb-1"
          >
            学习仪表盘
          </button>
        )}
        {onGraph && (
          <button
            onClick={onGraph}
            className="w-full text-xs text-center text-blue-600 hover:text-blue-800 py-1 rounded hover:bg-blue-50 transition mb-1"
          >
            知识图谱
          </button>
        )}
        <div className="text-xs text-slate-400 space-y-0.5">
          <p className="text-center font-medium">v2.0 · Agent Architecture</p>
          <p className="text-center">LangGraph + ChromaDB + Qwen</p>
        </div>
      </div>
    </aside>
  )
}
