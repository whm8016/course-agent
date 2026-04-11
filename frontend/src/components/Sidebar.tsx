import type { Course, Session, User } from '../types'
import CourseSelector from './CourseSelector'
import SessionList from './SessionList'

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
}: Props) {
  return (
    <aside className="w-64 h-full bg-white border-r border-slate-200 flex flex-col">
      <div className="px-4 py-5 border-b border-slate-100">
        <h1 className="text-base font-bold text-slate-800 flex items-center gap-2">
          <span className="text-xl">📚</span>
          课程学习 Agent
        </h1>
        <p className="text-xs text-slate-400 mt-1">LangGraph 多 Agent 编排</p>
      </div>

      <div className="p-3 border-b border-slate-100">
        <CourseSelector
          courses={courses}
          activeCourseId={activeCourseId}
          onSelect={onSelectCourse}
        />
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelect={onSelectSession}
          onCreate={onCreateSession}
          onDelete={onDeleteSession}
        />
      </div>

      <div className="p-4 border-t border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-600 font-medium truncate" title={user.display_name}>
            {user.display_name}
          </span>
          <button
            onClick={onLogout}
            className="text-xs text-slate-400 hover:text-red-500 transition"
          >
            退出
          </button>
        </div>
        <div className="text-xs text-slate-400 space-y-0.5">
          <p className="text-center font-medium">v2.0 · Agent Architecture</p>
          <p className="text-center">LangGraph + ChromaDB + Qwen</p>
        </div>
      </div>
    </aside>
  )
}
