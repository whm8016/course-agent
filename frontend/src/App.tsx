import { useEffect, useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import LoginPage from './components/LoginPage'
import AdminPage from './components/AdminPage'
import QuestionGeneratorPanel from './components/QuestionGeneratorPanel'
import { fetchCourses, fetchSessions, createSession, deleteSession } from './services/api'
import { isLoggedIn, getUser, logout } from './services/auth'
import type { Course, Session, User } from './types'
import './index.css'

export default function App() {
  const [user, setUser] = useState<User | null>(getUser())
  const [showAdmin, setShowAdmin] = useState(() => sessionStorage.getItem('_admin') === '1')
  const [courses, setCourses] = useState<Course[]>([])
  const [activeCourseId, setActiveCourseId] = useState<string>('')
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string>('')
  const [mainView, setMainView] = useState<'chat' | 'quiz'>('chat')

  const handleLogin = useCallback((u: User) => {
    setUser(u)
  }, [])

  const reloadCourses = useCallback(
    async (preserveActive: boolean = true) => {
      try {
        const list = await fetchCourses()
        setCourses(list)
        setLoadError('')
        if (list.length === 0) {
          setActiveCourseId('')
          return list
        }
        // 第一次加载、或当前选中的课程已被删，自动切到第一个
        setActiveCourseId((prev) => {
          if (!preserveActive) return list[0].id
          const stillExists = prev && list.some((c) => c.id === prev)
          return stillExists ? prev : list[0].id
        })
        return list
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : '加载课程失败'
        setLoadError(message)
        return [] as Course[]
      }
    },
    [],
  )

  useEffect(() => {
    if (!user) return
    void reloadCourses(false)
  }, [user, reloadCourses])

  // 只要存在 indexing 状态的课程，就每 5 秒轮询一次，等就绪后前端自动放开 RAG
  useEffect(() => {
    if (!user) return
    const hasIndexing = courses.some((c) => c.kb_status === 'indexing')
    if (!hasIndexing) return
    const t = setInterval(() => {
      void reloadCourses(true)
    }, 5000)
    return () => clearInterval(t)
  }, [user, courses, reloadCourses])

  const loadSessions = useCallback(async (courseId: string) => {
    try {
      const list = await fetchSessions(courseId)
      setSessions(list)
      if (list.length > 0) {
        setActiveSessionId(list[0].id)
      } else {
        setActiveSessionId(null)
      }
    } catch {
      setSessions([])
      setActiveSessionId(null)
    }
  }, [])

  useEffect(() => {
    if (activeCourseId && user) {
      loadSessions(activeCourseId)
    }
  }, [activeCourseId, loadSessions, user])

  const handleSelectCourse = (id: string) => {
    setActiveCourseId(id)
  }

  const handleCreateSession = async () => {
    if (!activeCourseId) return
    try {
      const session = await createSession(activeCourseId)
      setSessions((prev) => [session, ...prev])
      setActiveSessionId(session.id)
    } catch {
      // ignore
    }
  }

  const handleSessionCreated = useCallback((session: Session) => {
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
  }, [])

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
      if (activeSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id)
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null)
      }
    } catch {
      // ignore
    }
  }

  if (!isLoggedIn() || !user) {
    return <LoginPage onLogin={handleLogin} />
  }

  if (showAdmin && user.is_admin) {
    return (
      <AdminPage
        user={user}
        onBack={() => {
          sessionStorage.removeItem('_admin')
          setShowAdmin(false)
          void reloadCourses(true)
        }}
      />
    )
  }

  const activeCourse = courses.find((c) => c.id === activeCourseId)
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null

  return (
    <div className="flex h-screen bg-slate-50">
      <Sidebar
        courses={courses}
        activeCourseId={activeCourseId}
        onSelectCourse={handleSelectCourse}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
        user={user}
        onLogout={logout}
        onAdmin={user.is_admin ? () => { sessionStorage.setItem('_admin', '1'); setShowAdmin(true) } : undefined}
      />
      <main className="flex-1 h-full overflow-hidden flex flex-col">
        {/* 顶部视图切换栏 */}
        {activeCourse && (
          <div className="flex-none flex items-center gap-1 px-4 pt-3 pb-0 bg-slate-50 border-b border-slate-100">
            <button
              type="button"
              onClick={() => setMainView('chat')}
              className={`px-3 py-1.5 text-xs rounded-t-lg font-medium transition ${
                mainView === 'chat'
                  ? 'bg-white border border-b-white border-slate-200 text-indigo-600'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              💬 对话
            </button>
            <button
              type="button"
              onClick={() => setMainView('quiz')}
              className={`px-3 py-1.5 text-xs rounded-t-lg font-medium transition ${
                mainView === 'quiz'
                  ? 'bg-white border border-b-white border-slate-200 text-indigo-600'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              📝 出题
            </button>
          </div>
        )}
        <div className="flex-1 min-h-0 overflow-hidden">
          {activeCourse ? (
            <>
              <div
                className={
                  mainView === 'chat'
                    ? 'h-full min-h-0 flex flex-col overflow-hidden'
                    : 'hidden'
                }
              >
                <ChatWindow
                  courseId={activeCourseId}
                  courseName={`${activeCourse.icon} ${activeCourse.name}`}
                  sessionId={activeSessionId}
                  sessionMode={activeSession?.mode}
                  ragEnabled={Boolean(activeCourse.rag_enabled)}
                  kbStatus={activeCourse.kb_status ?? null}
                  onSessionCreated={handleSessionCreated}
                />
              </div>
              <div
                className={mainView === 'quiz' ? 'h-full min-h-0 overflow-hidden' : 'hidden'}
              >
                <QuestionGeneratorPanel kbName={activeCourseId} />
              </div>
            </>
          ) : loadError ? (
            <div className="flex items-center justify-center h-full text-red-500 px-8 text-center">
              {loadError}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400">
              加载中...
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
