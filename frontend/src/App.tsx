import { useEffect, useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import LoginPage from './components/LoginPage'
import { fetchCourses, fetchSessions, createSession, deleteSession } from './services/api'
import { isLoggedIn, getUser, logout } from './services/auth'
import type { Course, Session, User } from './types'
import './index.css'

export default function App() {
  const [user, setUser] = useState<User | null>(getUser())
  const [courses, setCourses] = useState<Course[]>([])
  const [activeCourseId, setActiveCourseId] = useState<string>('')
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string>('')

  const handleLogin = useCallback((u: User) => {
    setUser(u)
  }, [])

  useEffect(() => {
    if (!user) return
    fetchCourses()
      .then((list) => {
        setCourses(list)
        setLoadError('')
        if (list.length > 0) setActiveCourseId(list[0].id)
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : '加载课程失败'
        setLoadError(message)
      })
  }, [user])

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

  const activeCourse = courses.find((c) => c.id === activeCourseId)

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
      />
      <main className="flex-1 h-full overflow-hidden">
        {activeCourse ? (
          <ChatWindow
            courseId={activeCourseId}
            courseName={`${activeCourse.icon} ${activeCourse.name}`}
            sessionId={activeSessionId}
            onSessionCreated={handleSessionCreated}
          />
        ) : loadError ? (
          <div className="flex items-center justify-center h-full text-red-500 px-8 text-center">
            {loadError}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-400">
            加载中...
          </div>
        )}
      </main>
    </div>
  )
}
