import { useEffect, useState, useCallback, useRef, type ReactNode } from 'react'
import { FiMenu } from 'react-icons/fi'
import Sidebar from './components/layout/Sidebar'
import ChatWindow from './components/chat/ChatWindow'
import LoginPage from './components/pages/LoginPage'
import AdminPage from './components/pages/AdminPage'
import TeacherPage from './components/pages/TeacherPage'
import GraphPage from './components/pages/GraphPage'
import DashboardPanel from './components/pages/DashboardPanel'
import StudentStatsPage from './components/pages/StudentStatsPage'
import SkillKnowledgePage from './components/pages/SkillKnowledgePage'
import McpSettingsPage from './components/pages/McpSettingsPage'
import LlmProviderPage from './components/pages/LlmProviderPage'
import SearchProviderAdminPage from './components/pages/SearchProviderAdminPage'
import UserSearchSettingsPage from './components/pages/UserSearchSettingsPage'
import NotebookPage from './components/pages/NotebookPage'
import BotPage from './components/pages/BotPage'
import { fetchCourses, fetchSessions, createSession, deleteSession, fetchNotifications, markNotificationRead, joinCourseByCode, type BotNotificationItem } from './services/api'
import { isLoggedIn, getUser, logout } from './services/auth'
import type { Course, Session, User } from './types'
import './index.css'

export default function App() {
  const [user, setUser] = useState<User | null>(getUser())
  const [showAdmin, setShowAdmin] = useState(() => sessionStorage.getItem('_admin') === '1')
  const [showTeacher, setShowTeacher] = useState(() => sessionStorage.getItem('_teacher') === '1')
  const [showGraph, setShowGraph] = useState(false)
  const [showDashboard, setShowDashboard] = useState(false)
  const [showSkillKnowledge, setShowSkillKnowledge] = useState(false)
  const [showMcpSettings, setShowMcpSettings] = useState(false)
  const [showLlmProvider, setShowLlmProvider] = useState(false)
  const [showSearchAdmin, setShowSearchAdmin] = useState(false)
  const [showUserSearchSettings, setShowUserSearchSettings] = useState(false)
  const [showNotebook, setShowNotebook] = useState(false)
  const [showBots, setShowBots] = useState(false)
  const [showNotif, setShowNotif] = useState(false)
  const [notifList, setNotifList] = useState<BotNotificationItem[]>([])
  const [notifUnread, setNotifUnread] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [courses, setCourses] = useState<Course[]>([])
  const [activeCourseId, setActiveCourseId] = useState<string>('general')
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string>('')
  const [coursesLoading, setCoursesLoading] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; msg: string } | null>(null)
  const bootstrappedRef = useRef(false)

  const handleLogin = useCallback((u: User) => {
    setUser(u)
  }, [])

  // 启动时同步拦截 /join/{code}：暂存课程码到 sessionStorage + 立即清 URL。
  // 仅执行一次（ref 守卫），避免刷新/重渲染重复触发；未登录则等下方 effect 登录后消费。
  if (!bootstrappedRef.current) {
    bootstrappedRef.current = true
    const m = window.location.pathname.match(/^\/join\/([A-Za-z0-9\-]+)\/?$/)
    if (m) {
      sessionStorage.setItem('_pending_join_code', m[1])
      window.history.replaceState({}, '', '/')
    }
  }

  const loadNotifList = useCallback(async () => {
    try {
      const data = await fetchNotifications()
      setNotifList(data.notifications)
      setNotifUnread(data.unread_count)
    } catch {
      // ignore
    }
  }, [])

  // 轮询未读通知数（30s），驱动侧边栏铃铛 badge
  useEffect(() => {
    if (!user) return
    let active = true
    const poll = async () => {
      try {
        const data = await fetchNotifications()
        if (active) setNotifUnread(data.unread_count)
      } catch {
        // ignore
      }
    }
    void poll()
    const t = setInterval(poll, 30000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [user])

  const reloadCourses = useCallback(
    async (preserveActive: boolean = true) => {
      setCoursesLoading(true)
      try {
        const list = await fetchCourses()
        setCourses(list)
        setLoadError('')
        if (list.length === 0) {
          setActiveCourseId('general')
          return list
        }
        // 第一次加载、或当前选中的课程已被删，自动切到第一个
        setActiveCourseId((prev) => {
          if (!preserveActive) return list[0].id
          // 自由问答（general）是虚拟课程，不在选课列表里：切到它后刷新不被强制覆盖回第一门课
          if (prev === 'general') return 'general'
          const stillExists = prev && list.some((c) => c.id === prev)
          return stillExists ? prev : list[0].id
        })
        return list
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : '加载课程失败'
        setLoadError(message)
        return [] as Course[]
      } finally {
        setCoursesLoading(false)
      }
    },
    [],
  )

  useEffect(() => {
    if (!user) return
    void reloadCourses(false)
  }, [user, reloadCourses])

  // 登录后消费暂存的入课码（来自 /join/{code} 分享链接）：先移除防重复入课，再调接口 + 刷新课程列表
  useEffect(() => {
    if (!user) return
    const code = sessionStorage.getItem('_pending_join_code')
    if (!code) return
    sessionStorage.removeItem('_pending_join_code')
    void (async () => {
      try {
        const result = await joinCourseByCode(code)
        setToast({
          type: 'success',
          msg: result.already_enrolled ? `你已在课程《${result.name}》中` : `已成功加入课程《${result.name}》`,
        })
        await reloadCourses(false)
      } catch (e) {
        setToast({ type: 'error', msg: e instanceof Error ? e.message : '加入课程失败' })
      }
    })()
  }, [user, reloadCourses])

  // toast 自动消失（3.5s）
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 3500)
    return () => clearTimeout(t)
  }, [toast])

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

  const role = user.role ?? (user.is_admin ? 'admin' : 'student')
  const isTeacherOrAdmin = role === 'teacher' || role === 'admin'

  let overlay: ReactNode = null
  if (showTeacher && isTeacherOrAdmin) {
    overlay = (
      <TeacherPage
        user={user}
        onBack={() => {
          sessionStorage.removeItem('_teacher')
          setShowTeacher(false)
          void reloadCourses(true)
        }}
      />
    )
  } else if (showAdmin && role === 'admin') {
    overlay = (
      <AdminPage
        user={user}
        onBack={() => {
          sessionStorage.removeItem('_admin')
          setShowAdmin(false)
          void reloadCourses(true)
        }}
      />
    )
  } else if (showGraph) {
    overlay = <GraphPage onBack={() => setShowGraph(false)} />
  } else if (showDashboard) {
    if (isTeacherOrAdmin) {
      overlay = (
        <StudentStatsPage
          user={user}
          onBack={() => setShowDashboard(false)}
        />
      )
    } else {
      overlay = (
        <DashboardPanel
          onBack={() => setShowDashboard(false)}
          onGraph={() => { setShowDashboard(false); setShowGraph(true) }}
        />
      )
    }
  } else if (showSkillKnowledge) {
    overlay = (
      <SkillKnowledgePage
        courseId={activeCourseId}
        onBack={() => setShowSkillKnowledge(false)}
      />
    )
  } else if (showMcpSettings) {
    overlay = <McpSettingsPage onBack={() => setShowMcpSettings(false)} />
  } else if (showLlmProvider) {
    overlay = <LlmProviderPage onBack={() => setShowLlmProvider(false)} />
  } else if (showSearchAdmin) {
    overlay = <SearchProviderAdminPage onBack={() => setShowSearchAdmin(false)} />
  } else if (showUserSearchSettings) {
    overlay = <UserSearchSettingsPage onBack={() => setShowUserSearchSettings(false)} />
  } else if (showNotebook) {
    overlay = <NotebookPage onBack={() => setShowNotebook(false)} />
  } else if (showBots) {
    overlay = <BotPage onBack={() => setShowBots(false)} />
  } else if (showNotif) {
    overlay = (
      <NotificationPanel
        items={notifList}
        unread={notifUnread}
        onClose={() => setShowNotif(false)}
        onRead={async (id) => {
          try {
            await markNotificationRead(id)
            setNotifList((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)))
            setNotifUnread((u) => Math.max(0, u - 1))
          } catch {
            // ignore
          }
        }}
        onRefresh={() => void loadNotifList()}
      />
    )
  }

  const activeCourse = courses.find((c) => c.id === activeCourseId)
  // 自由问答（general）：未选课时的虚拟课程，不接入知识库，让用户不选课也能问答
  const GENERAL_COURSE: Course = {
    id: 'general',
    name: '自由问答',
    icon: '💬',
    description: '通用学习问答（未选课）',
    kb_status: null,
    rag_enabled: false,
    source: 'builtin',
  }
  const displayCourse = activeCourse ?? (activeCourseId === 'general' ? GENERAL_COURSE : null)
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null

  const closeSidebar = () => setSidebarOpen(false)

  return (
    <div className="flex h-screen bg-slate-50">
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 md:hidden"
          onClick={closeSidebar}
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transition-transform duration-200 md:relative md:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
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
          onAdmin={role === 'admin' ? () => { sessionStorage.setItem('_admin', '1'); setShowAdmin(true) } : undefined}
          onTeacher={isTeacherOrAdmin ? () => { sessionStorage.setItem('_teacher', '1'); setShowTeacher(true) } : undefined}
          onCoursesRefresh={() => void reloadCourses(false)}
          onDashboard={() => setShowDashboard(true)}
          onGraph={() => setShowGraph(true)}
          onSkillKnowledge={() => setShowSkillKnowledge(true)}
          onMcpSettings={() => setShowMcpSettings(true)}
          onLlmProvider={() => setShowLlmProvider(true)}
          onSearchAdmin={() => setShowSearchAdmin(true)}
          onUserSearchSettings={() => setShowUserSearchSettings(true)}
          onNotebook={() => setShowNotebook(true)}
          onBots={() => setShowBots(true)}
          onNotifications={() => {
            setShowNotif(true)
            void loadNotifList()
          }}
          notifUnread={notifUnread}
          onCloseMobile={closeSidebar}
        />
      </aside>
      <main className="flex-1 h-full overflow-hidden">
        {displayCourse ? (
          <ChatWindow
            courseId={activeCourseId}
            courseName={`${displayCourse.icon} ${displayCourse.name}`}
            sessionId={activeSessionId}
            sessionMode={activeSession?.mode}
            ragEnabled={Boolean(displayCourse.rag_enabled)}
            kbStatus={displayCourse.kb_status ?? null}
            onSessionCreated={handleSessionCreated}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        ) : (
          <div className="flex flex-col h-full">
            <div className="md:hidden px-3 py-3">
              <button onClick={() => setSidebarOpen(true)} className="p-1 text-slate-500 hover:text-slate-700 transition">
                <FiMenu size={20} />
              </button>
            </div>
            {loadError ? (
              <div className="flex items-center justify-center flex-1 text-red-500 px-8 text-center">
                {loadError}
              </div>
            ) : coursesLoading ? (
              <div className="flex items-center justify-center flex-1 text-slate-400">
                加载中...
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center flex-1 text-slate-400 gap-2">
                <span className="text-4xl">📭</span>
                <p className="text-sm">暂无课程</p>
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="md:hidden mt-2 px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                >
                  输入课程码加入
                </button>
                <p className="hidden md:block text-sm">请在左侧输入课程码加入课程</p>
              </div>
            )}
          </div>
        )}
      </main>
      {overlay && (
        <div className="fixed inset-0 z-50 bg-slate-50">{overlay}</div>
      )}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-[60] max-w-sm px-4 py-3 rounded-lg shadow-lg text-sm text-white flex items-start gap-3 ${
            toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          <span className="leading-relaxed">{toast.type === 'success' ? '✅' : '⚠️'}</span>
          <span className="flex-1 leading-relaxed">{toast.msg}</span>
          <button onClick={() => setToast(null)} className="text-white/80 hover:text-white shrink-0">
            ✕
          </button>
        </div>
      )}
    </div>
  )
}

function NotificationPanel({
  items,
  unread,
  onClose,
  onRead,
  onRefresh,
}: {
  items: BotNotificationItem[]
  unread: number
  onClose: () => void
  onRead: (id: string) => Promise<void>
  onRefresh: () => void
}) {
  return (
    <div className="h-full flex flex-col bg-slate-50">
      <header className="px-6 py-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm">
            ← 返回
          </button>
          <h1 className="text-lg font-semibold text-slate-800">🔔 通知</h1>
          {unread > 0 && (
            <span className="text-xs bg-red-500 text-white px-2 py-0.5 rounded-full">{unread}</span>
          )}
          <button onClick={onRefresh} className="text-xs text-slate-400 hover:text-slate-600">
            刷新
          </button>
        </div>
      </header>
      <div className="flex-1 overflow-y-auto p-6">
        {items.length === 0 ? (
          <div className="text-center text-slate-400 py-16">
            <span className="text-4xl">📭</span>
            <p className="text-sm mt-2">暂无通知</p>
            <p className="text-xs mt-1">Bot 定时提醒到点后会出现在这里</p>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-3">
            {items.map((n) => (
              <div
                key={n.id}
                className={`p-4 rounded-lg border ${
                  n.read ? 'bg-white border-slate-200' : 'bg-indigo-50/50 border-indigo-200'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-slate-400">
                    {n.bot_id ? `🤖 ${n.bot_id}` : '通知'} ·{' '}
                    {new Date(n.created_at * 1000).toLocaleString('zh-CN')}
                  </span>
                  {!n.read && (
                    <button
                      onClick={() => void onRead(n.id)}
                      className="text-xs text-indigo-600 hover:underline"
                    >
                      标为已读
                    </button>
                  )}
                </div>
                <p className="text-sm text-slate-700 whitespace-pre-wrap">{n.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
