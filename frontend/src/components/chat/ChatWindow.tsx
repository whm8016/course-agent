import { useState, useRef, useEffect, useCallback } from 'react'
import { FiSend, FiSquare, FiChevronDown, FiDatabase, FiGlobe, FiMenu, FiUploadCloud } from 'react-icons/fi'
import { BrainCircuit, MessageSquare, Microscope, PenLine, Zap } from 'lucide-react'
import MessageBubble from './MessageBubble'
import ImageUpload, { type PendingFile, isImage } from './ImageUpload'
import QuizConfigPanel, { DEFAULT_QUIZ_CONFIG, type QuizConfig } from '../quiz/QuizConfigPanel'
import { chatStream, uploadImage, fetchMessages, saveMessage, createSession, updateSessionMode, fetchLlmProfilesSelectable, fetchMyLlmProvider, deleteMyLlmProvider, requestAnswerNow } from '../../services/api'
import type { LlmProfileSelectable, UserProviderView } from '../../services/api'
import {
  connectQuestionGenerate,
  connectDeepResearch,
  type DeepResearchDepth,
  type DeepResearchMode,
  type DeepResearchSource,
} from '../../services/questionWs'
import type { Message, Session, SSEEvent, RagChunk, QuizData, ChatMode, HallucinationInfo, KBStatus, QuizQuestion, AttachmentInfo } from '../../types'

interface Props {
  courseId: string
  courseName: string
  sessionId: string | null
  sessionMode?: ChatMode
  ragEnabled?: boolean
  kbStatus?: KBStatus | null
  onSessionCreated: (session: Session) => void
  onOpenSidebar?: () => void
}

type ApiMessageRow = {
  role: string
  content: string
  msg_type?: string
  metadata?: Message['metadata'] & { attachments?: AttachmentInfo[] }
}

function rowToMessage(row: ApiMessageRow): Message {
  const mt = row.msg_type || 'text'
  return {
    role: row.role as 'user' | 'assistant',
    content: row.content,
    type: mt !== 'text' ? (mt as Message['type']) : undefined,
    metadata: row.metadata,
    attachments: row.metadata?.attachments,
  }
}

/** 与 MessageBubble 的 _thinkingSteps 同结构；必须每次 setState 用新引用，流式中才会重绘 */
function cloneThinkingSnapshot(steps: Message[]): Message[] {
  return steps.map((m) => ({ ...m, metadata: m.metadata ? { ...m.metadata } : undefined }))
}

// ---------- 能力定义 ----------
type CapValue = 'chat' | 'deep_solve' | 'quiz' | 'research'

interface CapDef {
  value: CapValue
  label: string
  description: string
  icon: React.ElementType
  chatMode: ChatMode
}

const CAPABILITIES: CapDef[] = [
  {
    value: 'chat',
    label: '对话',
    description: '通用问答，支持任意工具',
    icon: MessageSquare,
    chatMode: 'chat',
  },
  {
    value: 'deep_solve',
    label: '深度解题',
    description: '多步推理与问题求解',
    icon: BrainCircuit,
    chatMode: 'deep_solve',
  },
  {
    value: 'quiz',
    label: '出题',
    description: '自动生成并校验题目',
    icon: PenLine,
    chatMode: 'quiz',
  },
  {
    value: 'research',
    label: '深度研究',
    description: '全面多角度研究报告',
    icon: Microscope,
    chatMode: 'research',
  },
]

// ---------- QAPair → QuizQuestion ----------
interface QAPairRaw {
  question?: string
  correct_answer?: string
  explanation?: string
  question_type?: string
  options?: Record<string, string> | null
  concentration?: string
  difficulty?: string
}

function toQuizQuestion(qa: QAPairRaw): QuizQuestion {
  const opts = qa.options
  const type = (qa.question_type ?? '').toLowerCase()

  if (type === 'true_false') {
    return {
      question: qa.question ?? '',
      options: ['A. 正确', 'B. 错误'],
      answer:
        (qa.correct_answer ?? '').toUpperCase().startsWith('A') ||
        ['true', '正确', '对', 'yes'].includes((qa.correct_answer ?? '').toLowerCase())
          ? 'A'
          : 'B',
      explanation: qa.explanation ?? '',
    }
  }

  if (opts && typeof opts === 'object' && Object.keys(opts).length > 0) {
    const options = Object.entries(opts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([letter, text]) => `${letter}. ${text}`)
    const answer = (qa.correct_answer ?? '').charAt(0).toUpperCase() || 'A'
    return {
      question: qa.question ?? '',
      options,
      answer,
      explanation: qa.explanation ?? '',
    }
  }

  return {
    question: qa.question ?? '',
    options: [],
    answer: qa.correct_answer ?? '',
    explanation: qa.explanation ?? '',
  }
}

// ---------- 出题流式过程小字 ----------
interface QuizTraceRow {
  text: string
  kind: 'status' | 'progress' | 'done' | 'error'
}

interface QuizStreamingBubbleProps {
  traces: QuizTraceRow[]
  questions: QuizQuestion[]
  done: boolean
  error: string
}

function QuizStreamingBubble({ traces, questions, done, error }: QuizStreamingBubbleProps) {
  const lastTrace = traces[traces.length - 1]
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[95%] md:max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm space-y-2">
        {/* 小字过程 */}
        {!done && !error && (
          <div className="space-y-1">
            {traces.map((t, i) => (
              <p
                key={i}
                className={`text-[11px] leading-relaxed ${
                  i === traces.length - 1
                    ? 'text-indigo-500 animate-pulse'
                    : 'text-slate-400'
                }`}
              >
                {t.kind === 'status' ? '⚙ ' : t.kind === 'progress' ? '⟳ ' : ''}{t.text}
              </p>
            ))}
          </div>
        )}
        {done && lastTrace && (
          <p className="text-[11px] text-slate-400">{lastTrace.text}</p>
        )}
        {error && (
          <p className="text-[11px] text-red-500">{error}</p>
        )}
        {/* 已生成的题目（流式逐题追加） */}
        {questions.length > 0 && (
          <p className="text-xs text-slate-500 font-medium">已生成 {questions.length} 道题目</p>
        )}
      </div>
    </div>
  )
}

export default function ChatWindow({
  courseId,
  courseName,
  sessionId,
  sessionMode,
  ragEnabled = false,
  kbStatus = null,
  onSessionCreated,
  onOpenSidebar,
}: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingStarted, setStreamingStarted] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  // "立即回答"：流式过程中让模型基于已有信息直接作答（不中断 SSE）
  const [turnId, setTurnId] = useState<string | null>(null)
  const [isAnswerNowPending, setIsAnswerNowPending] = useState(false)

  // 能力 & 工具
  const [activeCap, setActiveCap] = useState<CapValue>('chat')
  const [capMenuOpen, setCapMenuOpen] = useState(false)
  const [useKb, setUseKb] = useState(false)
  const [useWebSearch, setUseWebSearch] = useState(false)

  // 知识库开关：默认跟随课程 KB 就绪状态（ready 时默认开），用户可手动开关；
  // 仅在切换课程 / KB 状态变化时重置，同课程内尊重用户的手动选择。
  useEffect(() => {
    setUseKb(ragEnabled)
  }, [ragEnabled])

  // 知识库检索模式：mix（混合）/ naive（向量）/ local（实体），默认 mix；仅 chat 流式 rag 工具消费
  const [ragMode, setRagMode] = useState<'mix' | 'naive' | 'local'>('mix')

  // 模型供应商（对标 DeepTutor：用户对话时可临时切换 provider/model）
  const [llmProfiles, setLlmProfiles] = useState<LlmProfileSelectable[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState('')
  // 学生个人 provider（/llm/me）：已配置时覆盖平台默认，顶部以徽章替代 admin profile 下拉
  const [myProvider, setMyProvider] = useState<UserProviderView | null>(null)
  const hasMyProvider = !!(myProvider && (myProvider.binding || myProvider.text_model || myProvider.vision_binding || myProvider.vision_model))

  // 出题配置面板
  const [quizConfig, setQuizConfig] = useState<QuizConfig>({ ...DEFAULT_QUIZ_CONFIG })
  // 出题流式状态（用于在消息列表内渲染）
  const [quizStreaming, setQuizStreaming] = useState(false)
  const [quizTraces, setQuizTraces] = useState<QuizTraceRow[]>([])
  const [quizStreamQuestions, setQuizStreamQuestions] = useState<QuizQuestion[]>([])
  const [quizError, setQuizError] = useState('')

  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const dragCounterRef = useRef(0)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  // 是否"贴在底部"跟随新内容。用户上拉超出阈值即 false（停止自动滚动，允许定住看）；
  // 滚回底部或点"回到底部"按钮恢复 true。用 state 而非 ref，是为了驱动按钮显隐重渲染。
  const [isAtBottom, setIsAtBottom] = useState(true)
  const currentSessionRef = useRef<string | null>(sessionId)
  // 标记“下一次 sessionId 变化是内部首次创建会话”（非用户切换），供切换 effect 跳过中止与重拉
  const suppressSwitchRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const quizCloseRef = useRef<(() => void) | null>(null)
  const researchCloseRef = useRef<(() => void) | null>(null)
  const capMenuRef = useRef<HTMLDivElement>(null)
  const capMenuMobileRef = useRef<HTMLDivElement>(null)

  // 深度研究流式状态
  const [researchStreaming, setResearchStreaming] = useState(false)
  const [researchTraces, setResearchTraces] = useState<{ text: string; kind: 'status' | 'progress' | 'done' | 'error' }[]>([])
  const [researchError, setResearchError] = useState('')
  const [researchMode, setResearchMode] = useState<DeepResearchMode>('report')
  const [researchDepth, setResearchDepth] = useState<DeepResearchDepth>('standard')
  // 默认仅联网检索（深度研究天然需查最新进展）；课程知识库(kb)由用户主动勾选，
    // 避免用户没开 rag 却被强行接知识库（后端按 metadata.sources 映射 rag/web_search）
    const [researchSources, setResearchSources] = useState<DeepResearchSource[]>(['web'])

  const chatMode: ChatMode = CAPABILITIES.find((c) => c.value === activeCap)?.chatMode ?? 'chat'
  const isQuizMode = activeCap === 'quiz'
  const isResearchMode = activeCap === 'research'

  useEffect(() => {
    currentSessionRef.current = sessionId
  }, [sessionId])

  // 从 sessionMode 恢复能力
  useEffect(() => {
    if (sessionMode) {
      const cap = CAPABILITIES.find((c) => c.chatMode === sessionMode)
      if (cap) setActiveCap(cap.value)
    }
  }, [sessionMode, sessionId])

  // 加载模型供应商：admin 预配的 profile 池（下拉）+ 学生个人 provider（/llm/me，覆盖默认）
  useEffect(() => {
    fetchLlmProfilesSelectable()
      .then((data) => setLlmProfiles(data.profiles || []))
      .catch(() => { /* 静默：未配置时不阻塞对话 */ })
    fetchMyLlmProvider()
      .then((v) => setMyProvider(v))
      .catch(() => { /* 静默：未登录个人配置时不阻塞 */ })
  }, [])

  // 窗口重新聚焦时刷新个人 provider（从配置页清除后回对话页，徽章及时消失）
  useEffect(() => {
    const reload = () => {
      fetchMyLlmProvider()
        .then((v) => setMyProvider(v))
        .catch(() => {})
    }
    window.addEventListener('focus', reload)
    return () => window.removeEventListener('focus', reload)
  }, [])

  // 复原为平台默认：清除个人 provider（DELETE /llm/me），徽章立即消失、下拉回归
  const handleResetToDefault = async () => {
    if (!confirm('清除个人模型配置，回到平台默认？')) return
    try {
      await deleteMyLlmProvider()
      setMyProvider(null)
      setSelectedProfileId('')
    } catch {
      // 静默：失败时徽章保留，用户可去「我的模型配置」页重试
    }
  }

  useEffect(() => {
    if (suppressSwitchRef.current) {
      // 内部首次创建会话：发起方正在写入消息，跳过重拉历史（避免覆盖刚写入的消息）
      return
    }
    if (!sessionId) {
      setMessages([])
      return
    }
    let cancelled = false
    fetchMessages(sessionId)
      .then((rows) => {
        if (cancelled) return
        setMessages((rows as ApiMessageRow[]).map(rowToMessage))
      })
      .catch(() => {
        if (!cancelled) setMessages([])
      })
    return () => {
      cancelled = true
    }
  }, [sessionId])

  // 用户切换对话/课程：中止旧会话进行中的流并重置生成 UI，避免串到新会话。
  // 内部首次创建会话时 suppressSwitchRef 为 true，这里消费并跳过（同一对话初始化，无流可中止）。
  useEffect(() => {
    if (suppressSwitchRef.current) {
      suppressSwitchRef.current = false
      return
    }
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    quizCloseRef.current?.()
    quizCloseRef.current = null
    researchCloseRef.current?.()
    researchCloseRef.current = null
    setLoading(false)
    setStreamingStarted(false)
    setIsStopping(false)
    setQuizStreaming(false)
    setQuizTraces([])
    setQuizStreamQuestions([])
    setQuizError('')
    setResearchStreaming(false)
    setResearchTraces([])
    setResearchError('')
  }, [sessionId])

  // 关闭能力菜单（点外部）
  useEffect(() => {
    if (!capMenuOpen) return
    const handler = (e: MouseEvent) => {
      const target = e.target as Node
      const inDesktop = capMenuRef.current?.contains(target)
      const inMobile = capMenuMobileRef.current?.contains(target)
      if (!inDesktop && !inMobile) setCapMenuOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [capMenuOpen])

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    // 真正"贴底"（距底 < 40px）才跟随。阈值从 80 收窄：流式时用户想看的内容多在底部
    // 附近，80px 内一旦上拉就被瞬时拽回、定不住。改 40 后上拉一点即脱离自动跟随。
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setIsAtBottom((prev) => (prev !== near ? near : prev))
  }, [])

  // 流式新内容到达时，仅在用户贴底期间跟随。直接赋值 el.scrollTop（瞬时、只动本容器），
  // 不用 bottomRef.scrollIntoView——后者会连带滚动祖先链，且 smooth 在高频流式下排队
  // 动画与用户手动滚动打架，是"拉不动/定不住"的主要来源。
  useEffect(() => {
    if (!isAtBottom) return
    const el = scrollContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, loading, quizStreaming, quizTraces, isAtBottom])

  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
    setIsAtBottom(true)
  }, [])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
      quizCloseRef.current?.()
      researchCloseRef.current?.()
    }
  }, [])

  const handleFileSelect = (file: File) => {
    const kind: 'image' | 'doc' = isImage(file) ? 'image' : 'doc'
    const preview = kind === 'image' ? URL.createObjectURL(file) : ''
    setPendingFiles((prev) => [...prev, { file, preview, kind, name: file.name }])
  }

  // 粘贴：从剪贴板提取图片/文件，复用 handleFileSelect；纯文本放行默认行为
  const handlePaste = (e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData.items)
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter((f): f is File => f !== null)
    if (files.length === 0) return // 纯文本：不阻止默认粘贴
    e.preventDefault() // 含图片：阻止把图片当文本插入
    files.forEach(handleFileSelect)
  }

  // 拖拽：用计数器而非布尔，避免经过子元素时 dragenter/leave 反复触发导致闪烁
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!Array.from(e.dataTransfer.types).includes('Files')) return // 拖选文本不触发
    dragCounterRef.current += 1
    setIsDragging(true)
  }
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault() // 必须 preventDefault 才能触发 drop
    e.stopPropagation()
  }
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current -= 1
    if (dragCounterRef.current <= 0) {
      dragCounterRef.current = 0
      setIsDragging(false)
    }
  }
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current = 0
    setIsDragging(false)
    Array.from(e.dataTransfer.files).forEach(handleFileSelect)
  }

  const removeFile = (index: number) => {
    setPendingFiles((prev) => {
      const target = prev[index]
      if (target?.preview) URL.revokeObjectURL(target.preview)
      return prev.filter((_, i) => i !== index)
    })
  }

  const clearFiles = () => {
    pendingFiles.forEach((pf) => {
      if (pf.preview) URL.revokeObjectURL(pf.preview)
    })
    setPendingFiles([])
  }

  // 上传待发送附件（pendingFiles → uploadImage → AttachmentInfo[]），chat / quiz / research 共用。
  // 不在此处 try/catch：上传失败由调用方决定（chat/quiz/research 各自提示并中止）。
  const uploadPendingAttachments = useCallback(async (): Promise<{
    attachments: AttachmentInfo[]
    firstImagePreview: string | undefined
  }> => {
    const attachments: AttachmentInfo[] = []
    let firstImagePreview: string | undefined
    for (const pf of pendingFiles) {
      const result = await uploadImage(pf.file)
      attachments.push({
        type: pf.kind === 'image' ? 'image' : 'file',
        url: result.path,
        filename: pf.file.name,
        mime_type: pf.file.type,
      })
      if (pf.kind === 'image' && !firstImagePreview) firstImagePreview = pf.preview
    }
    return { attachments, firstImagePreview }
  }, [pendingFiles])

  // ---------- 出题 ----------
  const handleQuizStart = useCallback(async () => {
    const topic = (quizConfig.topic.trim() || input.trim())
    if (!topic || !courseId) return
    if (loading || quizStreaming) return

    let activeSessionId = currentSessionRef.current
    if (!activeSessionId) {
      try {
        const session = await createSession(courseId, `出题: ${topic.slice(0, 20)}`, 'quiz')
        activeSessionId = session.id
        currentSessionRef.current = session.id
        suppressSwitchRef.current = true
        onSessionCreated(session)
      } catch {
        // ignore
      }
    }

    const mySession = activeSessionId
    const update = (updater: (prev: Message[]) => Message[]) => {
      setMessages((prev) => (currentSessionRef.current === mySession ? updater(prev) : prev))
    }

    // 上传待发送图片（可选；失败则提示并中止，与 chat 行为一致）
    let attachments: AttachmentInfo[] = []
    let firstImagePreview: string | undefined
    if (pendingFiles.length > 0) {
      try {
        ({ attachments, firstImagePreview } = await uploadPendingAttachments())
      } catch {
        setQuizError('图片上传失败，请重试')
        setQuizStreaming(false)
        return
      }
    }

    // 先把「用户请求」推入消息列表
    const userMsg: Message = {
      role: 'user',
      content: `出题：${topic}（${quizConfig.count} 道，${quizConfig.difficulty || '自动难度'}，${quizConfig.questionType || '自动题型'}）${quizConfig.preference ? `，偏好：${quizConfig.preference}` : ''}`,
      image: firstImagePreview,
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    setIsAtBottom(true)
    setInput('')
    update((prev) => [...prev, userMsg])
    clearFiles()
    setQuizStreaming(true)
    setQuizTraces([{ text: '连接中…', kind: 'status' }])
    setQuizStreamQuestions([])
    setQuizError('')

    const collectedQuestions: QuizQuestion[] = []

    const reqParts = [topic]
    if (quizConfig.difficulty) reqParts.push(`难度:${quizConfig.difficulty}`)
    if (quizConfig.questionType) reqParts.push(`题型:${quizConfig.questionType}`)
    if (quizConfig.preference.trim()) reqParts.push(quizConfig.preference.trim())
    const requirement = reqParts.join('，')

    const close = connectQuestionGenerate(
      {
        course_id: courseId,
        question: requirement,
        language: 'zh',
        metadata: { count: quizConfig.count, requirement },
        attachments: attachments.length > 0 ? attachments : undefined,
      },
      {
        onOpen: () =>
          setQuizTraces((prev) => [...prev, { text: '已连接，出题中…', kind: 'status' }]),
        onMessage: (msg) => {
          const t = msg.type
          if (t === 'stage_start') {
            const stage = String(msg.stage ?? '')
            const label =
              stage === 'explore'
                ? '探索素材'
                : stage === 'plan'
                  ? '规划蓝图'
                  : stage === 'quiz'
                    ? '逐题生成'
                    : stage
            setQuizTraces((prev) => [...prev, { text: label + '…', kind: 'progress' }])
          } else if (t === 'thinking') {
            const c = String(msg.content ?? '').trim()
            if (c) setQuizTraces((prev) => [...prev, { text: c, kind: 'status' }])
          } else if (t === 'quiz_question') {
            const q = (msg as { question?: QAPairRaw }).question
            if (q) {
              const qq = toQuizQuestion(q)
              collectedQuestions.push(qq)
              setQuizStreamQuestions((prev) => [...prev, qq])
            }
          } else if (t === 'result') {
            const qs = (msg as { questions?: QAPairRaw[] }).questions
            if (Array.isArray(qs) && qs.length > 0 && collectedQuestions.length === 0) {
              qs.forEach((qa) => {
                const qq = toQuizQuestion(qa)
                collectedQuestions.push(qq)
                setQuizStreamQuestions((prev) => [...prev, qq])
              })
            }
          } else if (t === 'done') {
            setQuizTraces((prev) => [
              ...prev,
              { text: `生成完成，共 ${collectedQuestions.length} 道`, kind: 'done' },
            ])
            setQuizStreaming(false)
            const quizData: QuizData = { questions: [...collectedQuestions] }
            const assistantMsg: Message = {
              role: 'assistant',
              content: '',
              metadata: { quiz: quizData },
            }
            update((prev) => [...prev, assistantMsg])
            setQuizStreamQuestions([])
            setQuizTraces([])
            quizCloseRef.current?.()
          } else if (t === 'error') {
            setQuizError(String(msg.message ?? msg.content ?? '出题失败，请重试'))
            setQuizStreaming(false)
          }
        },
        onClose: () => {
          setQuizStreaming((v) => {
            if (v) setQuizError('连接意外关闭')
            return false
          })
        },
        onError: () => {
          setQuizError('WebSocket 连接失败，请确认后端已启动')
          setQuizStreaming(false)
        },
      },
    )
    quizCloseRef.current = close
  }, [quizConfig, input, courseId, loading, quizStreaming, onSessionCreated, pendingFiles, uploadPendingAttachments])

  // ---------- 深度研究 ----------
  const handleResearchStart = useCallback(async (topic: string) => {
    if (!topic.trim() || !courseId) return
    if (loading || researchStreaming) return

    let activeSessionId = currentSessionRef.current
    if (!activeSessionId) {
      try {
        const session = await createSession(courseId, `研究: ${topic.slice(0, 20)}`, 'research')
        activeSessionId = session.id
        currentSessionRef.current = session.id
        suppressSwitchRef.current = true
        onSessionCreated(session)
      } catch {
        // ignore
      }
    }

    const mySession = activeSessionId
    const update = (updater: (prev: Message[]) => Message[]) => {
      setMessages((prev) => (currentSessionRef.current === mySession ? updater(prev) : prev))
    }

    // 上传待发送图片（可选；失败则提示并中止，与 chat 行为一致）
    let attachments: AttachmentInfo[] = []
    let firstImagePreview: string | undefined
    if (pendingFiles.length > 0) {
      try {
        ({ attachments, firstImagePreview } = await uploadPendingAttachments())
      } catch {
        setResearchError('图片上传失败，请重试')
        setResearchStreaming(false)
        return
      }
    }

    const userMsg: Message = {
      role: 'user',
      content: topic,
      image: firstImagePreview,
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    setIsAtBottom(true)
    update((prev) => [...prev, userMsg])
    clearFiles()
    setResearchStreaming(true)
    setResearchTraces([{ text: '连接中…', kind: 'status' }])
    setResearchError('')
    let reportContent = ''

    const stageLabel = (stage: string) =>
      stage === 'rephrase'
        ? '理解主题'
        : stage === 'decompose'
          ? '分解子主题'
          : stage === 'research'
            ? '检索研究'
            : stage === 'reporting'
              ? '撰写报告'
              : stage

    const finishWithReport = () => {
      if (!reportContent) return
      setResearchStreaming(false)
      setResearchTraces([])
      update((prev) => [
        ...prev,
        { role: 'assistant', content: reportContent, metadata: { mode: 'research' } },
      ])
      if (activeSessionId) {
        saveMessage(activeSessionId, 'user', topic, 'text').catch(() => {})
        saveMessage(activeSessionId, 'assistant', reportContent, 'text', { mode: 'research' }).catch(() => {})
      }
    }

    const close = connectDeepResearch(
      {
        course_id: courseId,
        question: topic.trim(),
        language: 'zh',
        metadata: { mode: researchMode, depth: researchDepth, sources: researchSources },
        attachments: attachments.length > 0 ? attachments : undefined,
      },
      {
        onOpen: () =>
          setResearchTraces((prev) => [...prev, { text: '已连接，研究中…', kind: 'status' }]),
        onMessage: (msg) => {
          const t = msg.type
          if (t === 'stage_start') {
            setResearchTraces((prev) => [
              ...prev,
              { text: stageLabel(String(msg.stage ?? '')) + '…', kind: 'progress' },
            ])
          } else if (t === 'thinking') {
            const c = String(msg.content ?? '').trim()
            if (c) setResearchTraces((prev) => [...prev, { text: c, kind: 'status' }])
          } else if (t === 'token') {
            reportContent += String(msg.content ?? '')
          } else if (t === 'answer') {
            const c = String(msg.content ?? '')
            if (c) reportContent = c
          } else if (t === 'result') {
            const r = String(msg.report ?? '')
            if (r) reportContent = r
            finishWithReport()
          } else if (t === 'done') {
            finishWithReport()
            setResearchStreaming(false)
            researchCloseRef.current?.()
          } else if (t === 'error') {
            setResearchError(String(msg.message ?? msg.content ?? '研究失败，请重试'))
            setResearchStreaming(false)
          }
        },
        onClose: () => {
          setResearchStreaming((v) => {
            if (v) setResearchError('连接意外关闭')
            return false
          })
        },
        onError: () => {
          setResearchError('WebSocket 连接失败，请确认后端已启动')
          setResearchStreaming(false)
        },
      },
    )
    researchCloseRef.current = close
  }, [courseId, loading, researchStreaming, researchMode, researchDepth, researchSources, onSessionCreated, pendingFiles, uploadPendingAttachments])

  // ---------- 普通聊天 ----------
  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text && pendingFiles.length === 0) return
    if (loading) return

    // quiz 模式不走聊天，改为出题
    if (isQuizMode) {
      await handleQuizStart()
      return
    }

    // research 模式不走聊天，改为深度研究
    if (isResearchMode) {
      await handleResearchStart(text)
      setInput('')
      return
    }

    let activeSessionId = currentSessionRef.current
    if (!activeSessionId) {
      try {
        const title = (text || (pendingFiles.length ? '附件对话' : '新对话')).slice(0, 20)
        const session = await createSession(courseId, title, chatMode)
        activeSessionId = session.id
        currentSessionRef.current = session.id
        suppressSwitchRef.current = true
        onSessionCreated(session)
      } catch {
        // fall through
      }
    }

    // 本次请求所属会话；切换到其它会话后丢弃其 UI 写入，避免串台
    const mySession = activeSessionId
    const update = (updater: (prev: Message[]) => Message[]) => {
      setMessages((prev) => (currentSessionRef.current === mySession ? updater(prev) : prev))
    }

    let attachments: AttachmentInfo[] = []
    let firstImagePreview: string | undefined

    if (pendingFiles.length > 0) {
      try {
        ({ attachments, firstImagePreview } = await uploadPendingAttachments())
      } catch {
        return
      }
    }

    const userMsg: Message = {
      role: 'user',
      content: text || (attachments.length ? '请分析这些文件' : ''),
      image: firstImagePreview,
      attachments: attachments.length > 0 ? attachments : undefined,
    }

    setIsAtBottom(true)
    update((prev) => [...prev, userMsg])
    setInput('')
    clearFiles()
    setLoading(true)
    setIsStopping(false)
    setStreamingStarted(false)
    setTurnId(null)
    setIsAnswerNowPending(false)
    const controller = new AbortController()
    abortControllerRef.current = controller

    const history = messages.map((m) => ({ role: m.role, content: m.content }))

    const thinkingSteps: Message[] = []
    let answerContent = ''
    let ragChunks: RagChunk[] = []
    let quizData: QuizData | undefined
    let intent = ''
    let intentConfidence = 0
    let resolvedMode: ChatMode = chatMode
    let toolsUsed: string[] = []
    let retrieveMode = ''
    let retrieveStrategy = ''
    let hallucination: HallucinationInfo | undefined

    const streamResult = await chatStream(
      courseId,
      userMsg.content,
      history,
      undefined,
      activeSessionId || undefined,
      chatMode,
      controller.signal,
      (event: SSEEvent) => {
        switch (event.type) {
          case 'turn_started':
            // 后端下发本回合 turn_id，供"立即回答"按钮 POST /chat/answer_now
            setTurnId(event.turn_id || null)
            break
          case 'thinking_chunk': {
            // 找到当前 stage 对应的最后一个 thinking step，把 token 追加进去
            const chunkStage = event.stage
            const chunkToken = event.content || ''
            const lastIdx = [...thinkingSteps].map((s, i) => ({ s, i }))
              .filter(({ s }) => s.type === 'thinking' && s.metadata?.stage === chunkStage)
              .pop()?.i
            if (lastIdx !== undefined) {
              thinkingSteps[lastIdx] = {
                ...thinkingSteps[lastIdx],
                content: (thinkingSteps[lastIdx].content || '') + chunkToken,
              }
            } else {
              // stage_chunk 先于 thinking start 到达时兜底
              thinkingSteps.push({
                role: 'assistant',
                content: chunkToken,
                type: 'thinking',
                metadata: { stage: chunkStage, call_state: 'running', timestamp: Date.now() / 1000 },
              })
            }
            setStreamingStarted(true)
            {
              const snap = cloneThinkingSnapshot(thinkingSteps)
              update((prev) => {
                const last = prev[prev.length - 1]
                const withSteps = (base: Message) => {
                  const o = { ...base } as Message & { _thinkingSteps?: Message[] }
                  o._thinkingSteps = snap
                  return o
                }
                if (last?.role === 'assistant') return [...prev.slice(0, -1), withSteps(last)]
                return [...prev, withSteps({ role: 'assistant', content: '' })]
              })
            }
            break
          }
          case 'thinking':
            thinkingSteps.push({
              role: 'assistant',
              content: event.content || '',
              type: 'thinking',
              metadata: {
                stage: event.stage,
                call_state: event.call_state,
                timestamp: Date.now() / 1000,
              },
            })
            setStreamingStarted(true)
            {
              const snap = cloneThinkingSnapshot(thinkingSteps)
              update((prev) => {
                const last = prev[prev.length - 1]
                const withSteps = (base: Message) => {
                  const o = { ...base } as Message & { _thinkingSteps?: Message[] }
                  o._thinkingSteps = snap
                  return o
                }
                if (last?.role === 'assistant') {
                  return [...prev.slice(0, -1), withSteps(last)]
                }
                return [...prev, withSteps({ role: 'assistant', content: '' })]
              })
            }
            break
          case 'tool_call':
            thinkingSteps.push({
              role: 'assistant',
              content: '',
              type: 'tool_call',
              metadata: { tool: event.tool, toolInput: event.input as Record<string, unknown> },
            })
            setStreamingStarted(true)
            {
              const snap = cloneThinkingSnapshot(thinkingSteps)
              update((prev) => {
                const last = prev[prev.length - 1]
                const withSteps = (base: Message) => {
                  const o = { ...base } as Message & { _thinkingSteps?: Message[] }
                  o._thinkingSteps = snap
                  return o
                }
                if (last?.role === 'assistant') {
                  return [...prev.slice(0, -1), withSteps(last)]
                }
                return [...prev, withSteps({ role: 'assistant', content: '' })]
              })
            }
            break
          case 'tool_result':
            if (event.chunks) {
              ragChunks = event.chunks
              thinkingSteps.push({
                role: 'assistant',
                content: '',
                type: 'tool_result',
                metadata: { chunks: event.chunks },
              })
              setStreamingStarted(true)
              {
                const snap = cloneThinkingSnapshot(thinkingSteps)
                update((prev) => {
                  const last = prev[prev.length - 1]
                  const withSteps = (base: Message) => {
                    const o = { ...base } as Message & { _thinkingSteps?: Message[] }
                    o._thinkingSteps = snap
                    return o
                  }
                  if (last?.role === 'assistant') {
                    return [...prev.slice(0, -1), withSteps(last)]
                  }
                  return [...prev, withSteps({ role: 'assistant', content: '' })]
                })
              }
            }
            break
          case 'token':
            setStreamingStarted(true)
            answerContent += event.content || ''
            update((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                const o = { ...last, content: answerContent } as Message & { _thinkingSteps?: Message[] }
                if ((last as Message & { _thinkingSteps?: Message[] })._thinkingSteps) {
                  o._thinkingSteps = (last as Message & { _thinkingSteps: Message[] })._thinkingSteps
                }
                return [...prev.slice(0, -1), o]
              }
              return [...prev, { role: 'assistant', content: answerContent }]
            })
            break
          case 'answer':
            setStreamingStarted(true)
            answerContent = event.content || ''
            update((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                const o = { ...last, content: answerContent } as Message & { _thinkingSteps?: Message[] }
                if ((last as Message & { _thinkingSteps?: Message[] })._thinkingSteps) {
                  o._thinkingSteps = (last as Message & { _thinkingSteps: Message[] })._thinkingSteps
                }
                return [...prev.slice(0, -1), o]
              }
              return [...prev, { role: 'assistant', content: answerContent }]
            })
            break
          case 'quiz':
            quizData = event.quiz
            break
          case 'skill_output':
            if (event.content) {
              const skillTitle = (event as unknown as Record<string, unknown>).title as string || '补充'
              answerContent += `\n\n---\n**${skillTitle}**\n\n${event.content}`
              update((prev) => {
                const last = prev[prev.length - 1]
                if (last?.role === 'assistant') {
                  return [...prev.slice(0, -1), { ...last, content: answerContent }]
                }
                return prev
              })
            }
            break
          case 'done':
            intent = event.metadata?.intent || ''
            intentConfidence = event.metadata?.intent_confidence || 0
            resolvedMode = (event.metadata?.mode as ChatMode) || chatMode
            toolsUsed = event.metadata?.tools_used || []
            retrieveMode = event.metadata?.retrieve_mode || ''
            retrieveStrategy = event.metadata?.retrieve_strategy || ''
            hallucination = event.metadata?.hallucination
            break
          case 'error':
            answerContent = `出错了: ${event.content}`
            break
        }
      },
      (err) => { answerContent = `出错了: ${err}` },
      [
        ...(useKb ? ['rag'] : []),
        ...(useWebSearch ? ['web_search'] : []),
      ],
      selectedProfileId || undefined,
      attachments.length > 0 ? attachments : undefined,
      useKb ? ragMode : 'mix',
    )
    abortControllerRef.current = null

    const wasAborted = streamResult.aborted
    const displayContent = wasAborted
      ? answerContent
        ? `${answerContent}\n\n_（已停止生成）_`
        : '_（已停止生成，未产生回答）_'
      : answerContent

    const assistantMsg: Message = {
      role: 'assistant',
      content: displayContent,
      metadata: {
        intent,
        intent_confidence: intentConfidence || undefined,
        mode: resolvedMode,
        chunks: ragChunks.length > 0 ? ragChunks : undefined,
        quiz: quizData,
        tools_used: toolsUsed.length > 0 ? toolsUsed : undefined,
        retrieve_mode: retrieveMode || undefined,
        retrieve_strategy: retrieveStrategy || undefined,
        hallucination,
        stopped: wasAborted || undefined,
      },
    }
    // @ts-expect-error attach thinking steps for rendering
    assistantMsg._thinkingSteps = [...thinkingSteps]

    update((prev) => {
      const lastIsAssistant = prev.length > 0 && prev[prev.length - 1].role === 'assistant'
      if (lastIsAssistant) return [...prev.slice(0, -1), assistantMsg]
      return [...prev, assistantMsg]
    })

    if (activeSessionId && !displayContent.startsWith('出错了')) {
      try {
        await saveMessage(activeSessionId, 'user', userMsg.content, 'text', attachments.length > 0 ? { attachments } : undefined)
        await saveMessage(activeSessionId, 'assistant', displayContent, 'text', {
          intent,
          intent_confidence: intentConfidence || undefined,
          mode: resolvedMode,
          tools_used: toolsUsed,
          chunks: ragChunks.length > 0 ? ragChunks : undefined,
          quiz: quizData,
          retrieve_mode: retrieveMode || undefined,
          retrieve_strategy: retrieveStrategy || undefined,
          hallucination,
          stopped: wasAborted || undefined,
        })
      } catch {
        // persistence is best-effort
      }
    }

    setLoading(false)
    setIsStopping(false)
    setStreamingStarted(false)
    setIsAnswerNowPending(false)
  }, [
    input,
    pendingFiles,
    loading,
    messages,
    courseId,
    onSessionCreated,
    chatMode,
    useKb,
    ragMode,
    selectedProfileId,
    isQuizMode,
    handleQuizStart,
    isResearchMode,
    handleResearchStart,
  ])

  // "立即回答"：让正在思考的模型基于已有信息直接作答（不中断 SSE，答案仍走原流）
  const handleAnswerNow = useCallback(async () => {
    if (!turnId || isAnswerNowPending) return
    setIsAnswerNowPending(true)
    await requestAnswerNow(turnId)
  }, [turnId, isAnswerNowPending])

  const handleStop = () => {
    if (quizStreaming) {
      quizCloseRef.current?.()
      setQuizStreaming(false)
      return
    }
    if (researchStreaming) {
      researchCloseRef.current?.()
      setResearchStreaming(false)
      return
    }
    if (!loading) return
    setIsStopping(true)
    abortControllerRef.current?.abort()
  }

  const handleSelectCap = async (cap: CapValue) => {
    setActiveCap(cap)
    setCapMenuOpen(false)
    if (currentSessionRef.current) {
      const mode = CAPABILITIES.find((c) => c.value === cap)?.chatMode ?? 'chat'
      try {
        await updateSessionMode(currentSessionRef.current, mode)
      } catch {
        // keep local
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (isQuizMode) void handleQuizStart()
      else if (isResearchMode) void handleResearchStart(input.trim())
      else void handleSend()
    }
  }

  const isRunning = loading || quizStreaming || researchStreaming
  const activeCapDef = CAPABILITIES.find((c) => c.value === activeCap)!
  const CapIcon = activeCapDef.icon

  return (
    <div className="flex flex-col h-full">
      {/* 顶部标题栏 */}
      <div className="border-b border-slate-200 px-3 md:px-6 py-3 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <button onClick={onOpenSidebar} className="md:hidden p-1 -ml-1 text-slate-500 hover:text-slate-700 transition">
            <FiMenu size={20} />
          </button>
          <h1 className="text-base font-semibold text-slate-800 truncate">{courseName}</h1>
          {hasMyProvider ? (
            <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-indigo-100 text-indigo-700 inline-flex items-center gap-1 max-w-[260px]">
              <span
                className="truncate min-w-0"
                title={`个人配置生效（覆盖平台默认）\n对话：${myProvider?.binding || '?'} / ${myProvider?.text_model || '?'}${myProvider?.vision_model ? `\n视觉：${myProvider?.vision_binding || myProvider?.binding || '?'}/${myProvider?.vision_model}` : ''}\n（在「我的模型配置」页修改）`}
              >
                🎯 个人：{myProvider?.binding || '?'} / {myProvider?.text_model || '?'}
              </span>
              <button
                onClick={handleResetToDefault}
                title="清除个人配置，回到平台默认"
                className="shrink-0 px-1 rounded-full leading-none text-indigo-500 hover:bg-indigo-200 hover:text-red-600"
              >✕</button>
            </span>
          ) : llmProfiles.length > 0 ? (
            <select
              value={selectedProfileId}
              onChange={(e) => setSelectedProfileId(e.target.value)}
              className="text-[11px] rounded-md border border-slate-200 px-1.5 py-0.5 bg-white text-slate-600 focus:outline-none focus:border-indigo-400 max-w-[150px]"
              title="选择模型供应商（临时切换，留空=使用默认）"
            >
              {llmProfiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name || p.id}{p.active ? '（默认）' : ''}
                </option>
              ))}
            </select>
          ) : null}
          {ragEnabled ? (
            <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-green-100 text-green-700">
              RAG 就绪
            </span>
          ) : kbStatus === 'indexing' ? (
            <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700">
              知识库索引中…
            </span>
          ) : null}
        </div>
        <p className="text-xs text-slate-400 mt-0.5">多 Agent 编排 · RAG 知识检索 · 智能出题</p>
      </div>

      {/* 消息列表 */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-3 md:px-6 py-4 bg-slate-50/50"
      >
        {messages.length === 0 && !quizStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <div className="text-4xl mb-3">
              {activeCap === 'quiz' ? '📝' : activeCap === 'deep_solve' ? '🧠' : activeCap === 'research' ? '🔎' : '💬'}
            </div>
            <p className="text-base font-medium">
              {activeCap === 'quiz' ? '配置参数后开始出题' : '开始提问吧'}
            </p>
            <p className="text-xs mt-1">
              {activeCap === 'quiz'
                ? '填写知识点，点击"开始出题"'
                : '输入问题、要求出题或上传图片'}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            courseId={courseId}
            thinkingSteps={(msg as unknown as Record<string, unknown>)._thinkingSteps as Message[] | undefined}
            isStreaming={loading && i === messages.length - 1}
          />
        ))}
        {/* 出题流式进度气泡 */}
        {quizStreaming && (
          <QuizStreamingBubble
            traces={quizTraces}
            questions={quizStreamQuestions}
            done={false}
            error={quizError}
          />
        )}
        {/* 深度研究流式进度气泡 */}
        {researchStreaming && (
          <div className="flex justify-start mb-4">
            <div className="max-w-[95%] md:max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-indigo-200 text-slate-800 rounded-bl-md shadow-sm space-y-1.5">
              {researchTraces.map((t, i) => (
                <p
                  key={i}
                  className={`text-[11px] leading-relaxed ${
                    i === researchTraces.length - 1
                      ? 'text-indigo-500 animate-pulse'
                      : 'text-slate-400'
                  }`}
                >
                  {t.kind === 'status' ? '⚙ ' : t.kind === 'progress' ? '🔍 ' : ''}{t.text}
                </p>
              ))}
              {researchError && (
                <p className="text-[11px] text-red-500">{researchError}</p>
              )}
            </div>
          </div>
        )}
        {/* 普通聊天 loading */}
        {loading && !streamingStarted && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" />
              </div>
              <p className="text-xs text-slate-400 mt-2">
                {activeCap === 'research' ? '深度研究中…' : activeCap === 'deep_solve' ? '深度推理中…' : '思考中…'}
              </p>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
        {/* 用户上拉脱离底部时显示：sticky 钉在滚动容器视口右下，不随内容滚；
            点击滚回底部并恢复自动跟随。贴底期间（isAtBottom=true）不渲染，零干扰。 */}
        {!isAtBottom && (
          <div className="sticky bottom-4 z-10 flex justify-end pointer-events-none">
            <button
              type="button"
              onClick={scrollToBottom}
              className="pointer-events-auto mb-1 flex items-center gap-1 rounded-full bg-white border border-slate-200 shadow-md px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 hover:text-slate-800 transition"
              title="回到底部"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
              回到底部
            </button>
          </div>
        )}
      </div>

      {/* 底部输入区 */}
      <div
        className={`relative border-t border-slate-200 bg-white px-3 md:px-4 pt-2 md:pt-3 pb-3 md:pb-4 ${
          isDragging ? 'ring-2 ring-indigo-300' : ''
        }`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center border-2 border-dashed border-indigo-400 bg-indigo-50/70">
            <div className="flex flex-col items-center gap-1 text-indigo-500">
              <FiUploadCloud size={22} strokeWidth={1.6} />
              <span className="text-[13px] font-medium">拖放文件到此</span>
              <span className="text-[11px] text-indigo-400">图片 / 文档 / 代码</span>
            </div>
          </div>
        )}
        {/* 出题配置面板（quiz 模式时展开） */}
        {isQuizMode && (
          <QuizConfigPanel value={quizConfig} onChange={setQuizConfig} />
        )}

        {/* 深度研究配置条（research 模式时展开） */}
        {isResearchMode && (
          <div className="flex flex-wrap gap-2 mb-2 px-1 py-2 rounded-xl bg-slate-50 border border-slate-200 text-xs">
            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-slate-400 shrink-0">模式</span>
              {(
                [
                  { v: 'report', label: '研究报告' },
                  { v: 'notes', label: '学习笔记' },
                  { v: 'comparison', label: '对比分析' },
                  { v: 'learning_path', label: '学习路径' },
                ] as { v: DeepResearchMode; label: string }[]
              ).map(({ v, label }) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setResearchMode(v)}
                  className={`px-2 py-0.5 rounded-full border transition ${
                    researchMode === v
                      ? 'bg-indigo-100 border-indigo-400 text-indigo-700 font-medium'
                      : 'border-slate-200 text-slate-500 hover:border-slate-300'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="w-px bg-slate-200 self-stretch hidden sm:block" />

            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-slate-400 shrink-0">深度</span>
              {(
                [
                  { v: 'quick', label: '快速' },
                  { v: 'standard', label: '标准' },
                  { v: 'deep', label: '深入' },
                ] as { v: DeepResearchDepth; label: string }[]
              ).map(({ v, label }) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setResearchDepth(v)}
                  className={`px-2 py-0.5 rounded-full border transition ${
                    researchDepth === v
                      ? 'bg-indigo-100 border-indigo-400 text-indigo-700 font-medium'
                      : 'border-slate-200 text-slate-500 hover:border-slate-300'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="w-px bg-slate-200 self-stretch hidden sm:block" />

            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-slate-400 shrink-0">资料</span>
              {(
                [
                  { v: 'kb', label: '知识库' },
                  { v: 'web', label: '网络' },
                  { v: 'papers', label: '论文' },
                ] as { v: DeepResearchSource; label: string }[]
              ).map(({ v, label }) => {
                const selected = researchSources.includes(v)
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() =>
                      setResearchSources((prev) =>
                        selected ? prev.filter((s) => s !== v) : [...prev, v],
                      )
                    }
                    className={`px-2 py-0.5 rounded-full border transition ${
                      selected
                        ? 'bg-indigo-100 border-indigo-400 text-indigo-700 font-medium'
                        : 'border-slate-200 text-slate-500 hover:border-slate-300'
                    }`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* 工具行 — 移动端独立成行 */}
        <div className="flex items-center gap-1.5 mb-2 md:hidden">
          <div className="relative shrink-0" ref={capMenuMobileRef}>
            <button
              type="button"
              onClick={() => setCapMenuOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border border-slate-200 bg-white text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition text-xs font-medium"
              title="切换能力"
            >
              <CapIcon size={14} strokeWidth={1.8} className="text-indigo-500" />
              {activeCapDef.label}
              <FiChevronDown size={12} />
            </button>
            {capMenuOpen && (
              <div className="absolute bottom-full left-0 mb-1.5 w-[220px] rounded-xl border border-slate-200 bg-white shadow-lg py-1.5 z-50">
                {CAPABILITIES.map((cap) => {
                  const Icon = cap.icon
                  const selected = activeCap === cap.value
                  return (
                    <button
                      key={cap.value}
                      type="button"
                      onClick={() => void handleSelectCap(cap.value)}
                      className={`flex w-full items-center gap-3 px-3.5 py-2 text-left transition-colors ${
                        selected ? 'bg-slate-50' : 'hover:bg-slate-50/60'
                      }`}
                    >
                      <Icon
                        size={15}
                        strokeWidth={1.6}
                        className={selected ? 'text-indigo-500' : 'text-slate-400'}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-medium text-slate-800">{cap.label}</div>
                        <div className="text-[11px] text-slate-400 truncate">{cap.description}</div>
                      </div>
                      {selected && (
                        <div className="h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </div>
          <button
            type="button"
            disabled={!ragEnabled}
            onClick={() => setUseKb((v) => !v)}
            title={ragEnabled ? '点击开关知识库检索' : '该课程暂无知识库'}
            className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-xl border text-xs transition ${
              useKb
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }${!ragEnabled ? ' cursor-not-allowed opacity-40' : ''}`}
          >
            <FiDatabase size={14} />
            <span className="text-[11px]">知识库</span>
          </button>
          {useKb && ragEnabled && (
            <select
              value={ragMode}
              onChange={(e) => setRagMode(e.target.value as 'mix' | 'naive' | 'local')}
              title="知识库检索模式"
              className="text-[11px] px-1.5 py-1.5 rounded-xl border border-slate-200 bg-white text-slate-600 focus:outline-none focus:border-indigo-400"
            >
              <option value="mix">混合</option>
              <option value="naive">向量</option>
              <option value="local">实体</option>
            </select>
          )}
          <button
            type="button"
            onClick={() => setUseWebSearch((v) => !v)}
            title="网络搜索"
            className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-xl border text-xs transition ${
              useWebSearch
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }`}
          >
            <FiGlobe size={14} />
            <span className="text-[11px]">搜索</span>
          </button>
          <ImageUpload files={pendingFiles} onSelect={handleFileSelect} onRemove={removeFile} />
        </div>

        <div className="flex items-end gap-2">
          {/* 桌面端工具按钮 — 移动端隐藏 */}
          <div className="relative shrink-0 hidden md:block" ref={capMenuRef}>
            <button
              type="button"
              onClick={() => setCapMenuOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl border border-slate-200 bg-white text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition text-xs font-medium"
              title="切换能力"
            >
              <CapIcon size={14} strokeWidth={1.8} className="text-indigo-500" />
              <span className="hidden sm:inline">{activeCapDef.label}</span>
              <FiChevronDown size={12} />
            </button>
            {capMenuOpen && (
              <div className="absolute bottom-full left-0 mb-1.5 w-[220px] rounded-xl border border-slate-200 bg-white shadow-lg py-1.5 z-50">
                {CAPABILITIES.map((cap) => {
                  const Icon = cap.icon
                  const selected = activeCap === cap.value
                  return (
                    <button
                      key={cap.value}
                      type="button"
                      onClick={() => void handleSelectCap(cap.value)}
                      className={`flex w-full items-center gap-3 px-3.5 py-2 text-left transition-colors ${
                        selected ? 'bg-slate-50' : 'hover:bg-slate-50/60'
                      }`}
                    >
                      <Icon
                        size={15}
                        strokeWidth={1.6}
                        className={selected ? 'text-indigo-500' : 'text-slate-400'}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-medium text-slate-800">{cap.label}</div>
                        <div className="text-[11px] text-slate-400 truncate">{cap.description}</div>
                      </div>
                      {selected && (
                        <div className="h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </div>
          <button
            type="button"
            disabled={!ragEnabled}
            onClick={() => setUseKb((v) => !v)}
            title={ragEnabled ? '点击开关知识库检索' : '该课程暂无知识库'}
            className={`hidden md:inline-flex items-center gap-1 px-2 py-2 rounded-xl border text-xs transition ${
              useKb
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }${!ragEnabled ? ' cursor-not-allowed opacity-40' : ''}`}
          >
            <FiDatabase size={14} />
            <span className="hidden sm:inline text-[11px]">知识库</span>
          </button>
          {useKb && ragEnabled && (
            <select
              value={ragMode}
              onChange={(e) => setRagMode(e.target.value as 'mix' | 'naive' | 'local')}
              title="知识库检索模式"
              className="hidden md:inline-block text-[11px] px-1.5 py-2 rounded-xl border border-slate-200 bg-white text-slate-600 focus:outline-none focus:border-indigo-400"
            >
              <option value="mix">混合</option>
              <option value="naive">向量</option>
              <option value="local">实体</option>
            </select>
          )}
          <button
            type="button"
            onClick={() => setUseWebSearch((v) => !v)}
            title="网络搜索"
            className={`hidden md:inline-flex items-center gap-1 px-2 py-2 rounded-xl border text-xs transition ${
              useWebSearch
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }`}
          >
            <FiGlobe size={14} />
            <span className="hidden sm:inline text-[11px]">搜索</span>
          </button>
          <div className="hidden md:block">
            <ImageUpload files={pendingFiles} onSelect={handleFileSelect} onRemove={removeFile} />
          </div>

          {/* 输入框 */}
          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder={
                isQuizMode
                  ? '输入知识点直接出题…'
                  : activeCap === 'deep_solve'
                  ? '输入题目，深度推理求解…'
                  : activeCap === 'research'
                  ? '输入研究主题…'
                  : '输入问题（Shift+Enter 换行）'
              }
              rows={1}
              disabled={isRunning}
              className="w-full resize-none rounded-xl border border-slate-200 px-3 md:px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition disabled:opacity-50"
              style={{ minHeight: '42px', maxHeight: '120px' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement
                target.style.height = 'auto'
                target.style.height = Math.min(target.scrollHeight, 120) + 'px'
              }}
            />
          </div>

          {/* 立即回答按钮：仅 chat/deep_solve 流式且有 turn_id 时显示（quiz/research 不支持） */}
          {isRunning && turnId && !isQuizMode && !isResearchMode && (
            <button
              onClick={handleAnswerNow}
              disabled={isAnswerNowPending}
              className="p-2.5 rounded-xl text-white disabled:opacity-40 disabled:cursor-not-allowed transition shrink-0 bg-amber-500 hover:bg-amber-600"
              title="立即回答（基于已有信息直接作答）"
            >
              <Zap size={17} />
            </button>
          )}

          {/* 发送 / 停止按钮 */}
          <button
            onClick={isRunning ? handleStop : () => {
              if (isQuizMode) void handleQuizStart()
              else if (isResearchMode) void handleResearchStart(input.trim())
              else void handleSend()
            }}
            disabled={
              isStopping ||
              (!isRunning &&
                !input.trim() &&
                pendingFiles.length === 0 &&
                !(isQuizMode && (quizConfig.topic.trim() || input.trim())))
            }
            className={`p-2.5 rounded-xl text-white disabled:opacity-40 disabled:cursor-not-allowed transition shrink-0 ${
              isRunning ? 'bg-rose-600 hover:bg-rose-700' : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
            title={isRunning ? '停止' : isQuizMode ? '开始出题' : '发送'}
          >
            {isRunning ? <FiSquare size={17} /> : isQuizMode ? <PenLine size={17} /> : <FiSend size={17} />}
          </button>
        </div>
      </div>
    </div>
  )
}
