import { useEffect, useState, useCallback, useRef, type ReactNode } from 'react'
import { Menu, Bell, Inbox, Bot, Briefcase, Settings, Presentation, Shield, LayoutDashboard, Network, Backpack, NotebookPen, Cpu, Plug, Search, SlidersHorizontal } from 'lucide-react'
import Sidebar from './components/layout/Sidebar'
import HubPage, { type HubItem } from './components/layout/HubPage'
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
import { Button, EmptyState, ToastViewport } from './components/ui'
import { toast } from './lib/toast'
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
  const [showWorkbench, setShowWorkbench] = useState(false)
  const [showSettingsHub, setShowSettingsHub] = useState(false)
  const [notifList, setNotifList] = useState<BotNotificationItem[]>([])
  const [notifUnread, setNotifUnread] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [courses, setCourses] = useState<Course[]>([])
  const [activeCourseId, setActiveCourseId] = useState<string>('general')
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string>('')
  const [coursesLoading, setCoursesLoading] = useState(false)
  const bootstrappedRef = useRef(false)

  const handleLogin = useCallback((u: User) => {
    setUser(u)
  }, [])

  // 启动时同步拦截 /join/{code}：暂存课程码到 sessionStorage + 立即清 URL。
  // 仅执行一次（ref 守卫），避免刷新/重渲染重复触发；未登录则等下方 effect 登录后消费。
  if (!bootstrappedRef.current) {
    bootstrappedRef.current = true
    const m = window.location.pathname.match(/^\/join\/([A-Za-z0-9-]+)\/?$/)
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
        toast.success(
          result.already_enrolled
            ? `你已在课程《${result.name}》中`
            : `已成功加入课程《${result.name}》`,
        )
        await reloadCourses(false)
      } catch (e) {
        toast.error(e instanceof Error ? e.message : '加入课程失败')
      }
    })()
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

  const role = user.role ?? (user.is_admin ? 'admin' : 'student')
  const isTeacherOrAdmin = role === 'teacher' || role === 'admin'

  const workbenchItems: HubItem[] = [
    {
      icon: LayoutDashboard,
      label: isTeacherOrAdmin ? '学生学情统计' : '学习仪表盘',
      desc: isTeacherOrAdmin ? '学生答题、掌握度与风险预警' : '你的学习数据与进度概览',
      onClick: () => { setShowWorkbench(false); setShowDashboard(true) },
    },
    {
      icon: Network,
      label: '知识图谱',
      desc: '可视化课程知识点关联',
      onClick: () => { setShowWorkbench(false); setShowGraph(true) },
    },
    {
      icon: Backpack,
      label: '技能知识包',
      desc: '管理课程技能与知识点',
      onClick: () => { setShowWorkbench(false); setShowSkillKnowledge(true) },
    },
    {
      icon: NotebookPen,
      label: '题目笔记本',
      desc: '收藏题目与错题本',
      onClick: () => { setShowWorkbench(false); setShowNotebook(true) },
    },
    ...(isTeacherOrAdmin ? [{
      icon: Presentation,
      label: '教师工作台',
      desc: '知识库、学生学情、课码分享',
      onClick: () => { setShowWorkbench(false); sessionStorage.setItem('_teacher', '1'); setShowTeacher(true) },
    }] : []),
    ...(role === 'admin' ? [{
      icon: Shield,
      label: '管理后台',
      desc: '用户、课程、模型与审批',
      onClick: () => { setShowWorkbench(false); sessionStorage.setItem('_admin', '1'); setShowAdmin(true) },
    }] : []),
  ]

  const settingsItems: HubItem[] = [
    {
      icon: Cpu,
      label: role === 'admin' ? '模型供应商' : '我的模型配置',
      desc: role === 'admin' ? '平台模型与 API 凭证' : '个人模型配置（覆盖平台默认）',
      onClick: () => { setShowSettingsHub(false); setShowLlmProvider(true) },
    },
    {
      icon: Plug,
      label: 'MCP 工具',
      desc: role === 'admin' ? '部署与配置 MCP 服务器' : '选择启用的 MCP 工具',
      onClick: () => { setShowSettingsHub(false); setShowMcpSettings(true) },
    },
    ...(role === 'admin' ? [{
      icon: Search,
      label: '搜索引擎（默认）',
      desc: '配置平台默认 Web 搜索引擎',
      onClick: () => { setShowSettingsHub(false); setShowSearchAdmin(true) },
    }] : []),
    {
      icon: SlidersHorizontal,
      label: '我的搜索设置',
      desc: '个人搜索偏好',
      onClick: () => { setShowSettingsHub(false); setShowUserSearchSettings(true) },
    },
    ...(isTeacherOrAdmin ? [{
      icon: Bot,
      label: 'Bot 管理',
      desc: '定时提醒与 Bot 任务',
      onClick: () => { setShowSettingsHub(false); setShowBots(true) },
    }] : []),
    {
      icon: Bell,
      label: '通知',
      desc: 'Bot 提醒与系统消息',
      badge: notifUnread,
      onClick: () => { setShowSettingsHub(false); setShowNotif(true); void loadNotifList() },
    },
  ]

  let overlay: ReactNode = null
  if (showWorkbench) {
    overlay = (
      <HubPage
        title="工作台"
        icon={Briefcase}
        subtitle="学习与管理工作入口"
        items={workbenchItems}
        onBack={() => setShowWorkbench(false)}
      />
    )
  } else if (showSettingsHub) {
    overlay = (
      <HubPage
        title="设置"
        icon={Settings}
        subtitle="模型、检索与系统配置"
        items={settingsItems}
        onBack={() => setShowSettingsHub(false)}
      />
    )
  } else if (showTeacher && isTeacherOrAdmin) {
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
    <div className="flex h-screen bg-canvas">
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
          onCoursesRefresh={() => void reloadCourses(false)}
          onWorkbench={() => setShowWorkbench(true)}
          onSettings={() => setShowSettingsHub(true)}
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
            indexBackends={displayCourse.index_backends}
            onSessionCreated={handleSessionCreated}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        ) : (
          <div className="flex flex-col h-full bg-canvas">
            <div className="md:hidden px-3 py-3">
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1 text-ink-soft hover:text-ink transition"
                aria-label="打开侧边栏"
              >
                <Menu size={20} strokeWidth={1.5} />
              </button>
            </div>
            {loadError ? (
              <div className="flex items-center justify-center flex-1 text-danger-fg px-8 text-center">
                {loadError}
              </div>
            ) : coursesLoading ? (
              <div className="flex items-center justify-center flex-1 text-muted">加载中...</div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center gap-4">
                <EmptyState icon={Inbox} title="暂无课程" hint="请在左侧输入课程码加入课程" />
                <Button
                  variant="primary"
                  className="md:hidden"
                  onClick={() => setSidebarOpen(true)}
                >
                  输入课程码加入
                </Button>
              </div>
            )}
          </div>
        )}
      </main>
      {overlay && <div className="fixed inset-0 z-50 bg-canvas">{overlay}</div>}
      <ToastViewport />
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
    <div className="h-full flex flex-col bg-canvas">
      <header className="px-6 py-4 bg-surface border-b border-line flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onClose} className="text-ink-soft hover:text-ink text-sm transition">
            ← 返回
          </button>
          <h1 className="font-serif text-lg text-ink flex items-center gap-2">
            <Bell size={18} strokeWidth={1.5} className="text-ink-soft" />
            通知
          </h1>
          {unread > 0 && (
            <span className="text-xs bg-danger-fg text-white px-2 py-0.5 rounded-full">
              {unread}
            </span>
          )}
          <button onClick={onRefresh} className="text-xs text-muted hover:text-ink-soft transition">
            刷新
          </button>
        </div>
      </header>
      <div className="flex-1 overflow-y-auto p-6">
        {items.length === 0 ? (
          <EmptyState icon={Inbox} title="暂无通知" hint="Bot 定时提醒到点后会出现在这里" />
        ) : (
          <div className="max-w-2xl mx-auto space-y-3">
            {items.map((n) => (
              <div
                key={n.id}
                className={`p-4 rounded-[var(--radius)] border ${
                  n.read ? 'bg-surface border-line' : 'bg-info-bg border-line'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted flex items-center gap-1">
                    {n.bot_id && <Bot size={12} strokeWidth={1.5} />}
                    {n.bot_id ?? '通知'}
                    {' · '}
                    {new Date(n.created_at * 1000).toLocaleString('zh-CN')}
                  </span>
                  {!n.read && (
                    <button
                      onClick={() => void onRead(n.id)}
                      className="text-xs text-ink-soft hover:text-ink transition"
                    >
                      标为已读
                    </button>
                  )}
                </div>
                <p className="text-sm text-ink whitespace-pre-wrap">{n.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
