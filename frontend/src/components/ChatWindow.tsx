import { useState, useRef, useEffect, useCallback } from 'react'
import { FiSend, FiSquare, FiChevronDown, FiDatabase, FiGlobe } from 'react-icons/fi'
import { BrainCircuit, MessageSquare, Microscope, PenLine } from 'lucide-react'
import MessageBubble from './MessageBubble'
import ImageUpload from './ImageUpload'
import QuizConfigPanel, { DEFAULT_QUIZ_CONFIG, type QuizConfig } from './QuizConfigPanel'
import { chatStream, uploadImage, fetchMessages, saveMessage, createSession, updateSessionMode } from '../services/api'
import { connectQuestionGenerate } from '../services/questionWs'
import type { Message, Session, SSEEvent, RagChunk, QuizData, ChatMode, GuardrailInfo, HallucinationInfo, KBStatus, QuizQuestion } from '../types'

interface Props {
  courseId: string
  courseName: string
  sessionId: string | null
  sessionMode?: ChatMode
  ragEnabled?: boolean
  kbStatus?: KBStatus | null
  onSessionCreated: (session: Session) => void
}

type ApiMessageRow = {
  role: string
  content: string
  msg_type?: string
  metadata?: Message['metadata']
}

function rowToMessage(row: ApiMessageRow): Message {
  const mt = row.msg_type || 'text'
  return {
    role: row.role as 'user' | 'assistant',
    content: row.content,
    type: mt !== 'text' ? (mt as Message['type']) : undefined,
    metadata: row.metadata,
  }
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
      <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm space-y-2">
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
}: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingStarted, setStreamingStarted] = useState(false)
  const [isStopping, setIsStopping] = useState(false)

  // 能力 & 工具
  const [activeCap, setActiveCap] = useState<CapValue>('chat')
  const [capMenuOpen, setCapMenuOpen] = useState(false)
  const [useKb, setUseKb] = useState(false)
  const [useWebSearch, setUseWebSearch] = useState(false)

  // 出题配置面板
  const [quizConfig, setQuizConfig] = useState<QuizConfig>({ ...DEFAULT_QUIZ_CONFIG })
  // 出题流式状态（用于在消息列表内渲染）
  const [quizStreaming, setQuizStreaming] = useState(false)
  const [quizTraces, setQuizTraces] = useState<QuizTraceRow[]>([])
  const [quizStreamQuestions, setQuizStreamQuestions] = useState<QuizQuestion[]>([])
  const [quizError, setQuizError] = useState('')

  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const isUserNearBottomRef = useRef(true)
  const currentSessionRef = useRef<string | null>(sessionId)
  const abortControllerRef = useRef<AbortController | null>(null)
  const quizCloseRef = useRef<(() => void) | null>(null)
  const capMenuRef = useRef<HTMLDivElement>(null)

  const chatMode: ChatMode = CAPABILITIES.find((c) => c.value === activeCap)?.chatMode ?? 'chat'
  const isQuizMode = activeCap === 'quiz'

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

  useEffect(() => {
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

  // 关闭能力菜单（点外部）
  useEffect(() => {
    if (!capMenuOpen) return
    const handler = (e: MouseEvent) => {
      if (capMenuRef.current && !capMenuRef.current.contains(e.target as Node)) {
        setCapMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [capMenuOpen])

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    isUserNearBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  useEffect(() => {
    if (isUserNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: loading ? 'auto' : 'smooth' })
    }
  }, [messages, loading, quizStreaming, quizTraces])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
      quizCloseRef.current?.()
    }
  }, [])

  const handleImageSelect = (file: File) => {
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
  }

  const clearImage = () => {
    setImageFile(null)
    if (imagePreview) URL.revokeObjectURL(imagePreview)
    setImagePreview(null)
  }

  // ---------- 出题 ----------
  const handleQuizStart = useCallback(async () => {
    if (!quizConfig.topic.trim() || !courseId) return
    if (loading || quizStreaming) return

    let activeSessionId = currentSessionRef.current
    if (!activeSessionId) {
      try {
        const session = await createSession(courseId, `出题: ${quizConfig.topic.slice(0, 20)}`, 'quiz')
        activeSessionId = session.id
        currentSessionRef.current = session.id
        onSessionCreated(session)
      } catch {
        // ignore
      }
    }

    // 先把「用户请求」推入消息列表
    const userMsg: Message = {
      role: 'user',
      content: `出题：${quizConfig.topic}（${quizConfig.count} 道，${quizConfig.difficulty || '自动难度'}，${quizConfig.questionType || '自动题型'}）${quizConfig.preference ? `，偏好：${quizConfig.preference}` : ''}`,
    }
    isUserNearBottomRef.current = true
    setMessages((prev) => [...prev, userMsg])
    setQuizStreaming(true)
    setQuizTraces([{ text: '连接中…', kind: 'status' }])
    setQuizStreamQuestions([])
    setQuizError('')

    const collectedQuestions: QuizQuestion[] = []

    const close = connectQuestionGenerate(
      {
        kb_name: courseId,
        count: quizConfig.count,
        language: 'zh',
        requirement: {
          knowledge_point: quizConfig.topic.trim(),
          preference: quizConfig.preference.trim() || undefined,
          difficulty: quizConfig.difficulty || undefined,
          question_type: quizConfig.questionType || undefined,
        },
      },
      {
        onOpen: () =>
          setQuizTraces((prev) => [...prev, { text: '已连接，出题中…', kind: 'status' }]),
        onMessage: (msg) => {
          const t = msg.type
          if (t === 'status') {
            setQuizTraces((prev) => [
              ...prev,
              { text: String(msg.content ?? ''), kind: 'status' },
            ])
          } else if (t === 'progress') {
            const stage = String(msg.stage ?? '')
            const cur = Number(msg.current ?? 0)
            const tot = Number(msg.total ?? 0)
            let text = ''
            if (stage === 'ideation') text = `分析知识点，生成模板 ${cur}/${tot}…`
            else if (stage === 'generation') text = `正在生成第 ${cur}/${tot} 道题…`
            else if (stage === 'complete') text = `生成完成，共 ${msg.completed ?? cur} 道`
            if (text) setQuizTraces((prev) => [...prev, { text, kind: 'progress' }])
          } else if (t === 'result') {
            const qa = (msg as { qa_pair?: QAPairRaw }).qa_pair
            if (qa) {
              const q = toQuizQuestion(qa)
              collectedQuestions.push(q)
              setQuizStreamQuestions((prev) => [...prev, q])
            }
          } else if (t === 'complete') {
            setQuizTraces((prev) => [
              ...prev,
              { text: `生成完成，共 ${collectedQuestions.length} 道`, kind: 'done' },
            ])
            setQuizStreaming(false)
            // 将结果以 assistant 消息写入列表
            const quizData: QuizData = { questions: [...collectedQuestions] }
            const assistantMsg: Message = {
              role: 'assistant',
              content: '',
              metadata: { quiz: quizData },
            }
            setMessages((prev) => [...prev, assistantMsg])
            setQuizStreamQuestions([])
            setQuizTraces([])
            quizCloseRef.current?.()
          } else if (t === 'error') {
            setQuizError(String(msg.content ?? '出题失败，请重试'))
            setQuizStreaming(false)
          }
        },
        onClose: () => {
          // onComplete fires first; this handles unexpected close
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
  }, [quizConfig, courseId, loading, quizStreaming, onSessionCreated])

  // ---------- 普通聊天 ----------
  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text && !imageFile) return
    if (loading) return

    // quiz 模式不走聊天，改为出题
    if (isQuizMode) {
      if (text) setQuizConfig((c) => ({ ...c, topic: text }))
      await handleQuizStart()
      setInput('')
      return
    }

    let activeSessionId = currentSessionRef.current
    if (!activeSessionId) {
      try {
        const title = (text || '图片分析').slice(0, 20) || '新对话'
        const session = await createSession(courseId, title, chatMode)
        activeSessionId = session.id
        currentSessionRef.current = session.id
        onSessionCreated(session)
      } catch {
        // fall through
      }
    }

    let uploadedPath: string | undefined
    let displayUrl: string | undefined

    if (imageFile) {
      try {
        const result = await uploadImage(imageFile)
        uploadedPath = result.path
        displayUrl = imagePreview || undefined
      } catch {
        return
      }
    }

    const userMsg: Message = {
      role: 'user',
      content: text || '请分析这张图片',
      image: displayUrl,
    }

    isUserNearBottomRef.current = true
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    clearImage()
    setLoading(true)
    setIsStopping(false)
    setStreamingStarted(false)
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
    let guardrail: GuardrailInfo | undefined
    let hallucination: HallucinationInfo | undefined

    const effectiveRagEnabled = ragEnabled || useKb

    const streamResult = await chatStream(
      courseId,
      userMsg.content,
      history,
      uploadedPath,
      activeSessionId || undefined,
      chatMode,
      controller.signal,
      (event: SSEEvent) => {
        switch (event.type) {
          case 'thinking':
            thinkingSteps.push({ role: 'assistant', content: event.content || '', type: 'thinking' })
            break
          case 'tool_call':
            thinkingSteps.push({
              role: 'assistant',
              content: '',
              type: 'tool_call',
              metadata: { tool: event.tool, toolInput: event.input as Record<string, unknown> },
            })
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
            }
            break
          case 'token':
            setStreamingStarted(true)
            answerContent += event.content || ''
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { ...last, content: answerContent }]
              }
              return [...prev, { role: 'assistant', content: answerContent }]
            })
            break
          case 'answer':
            setStreamingStarted(true)
            answerContent = event.content || ''
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { ...last, content: answerContent }]
              }
              return [...prev, { role: 'assistant', content: answerContent }]
            })
            break
          case 'quiz':
            quizData = event.quiz
            break
          case 'done':
            intent = event.metadata?.intent || ''
            intentConfidence = event.metadata?.intent_confidence || 0
            resolvedMode = (event.metadata?.mode as ChatMode) || chatMode
            toolsUsed = event.metadata?.tools_used || []
            retrieveMode = event.metadata?.retrieve_mode || ''
            retrieveStrategy = event.metadata?.retrieve_strategy || ''
            guardrail = event.metadata?.guardrail
            hallucination = event.metadata?.hallucination
            break
          case 'error':
            answerContent = `出错了: ${event.content}`
            break
        }
      },
      (err) => { answerContent = `出错了: ${err}` },
      effectiveRagEnabled,
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
        guardrail,
        hallucination,
        stopped: wasAborted || undefined,
      },
    }
    // @ts-expect-error attach thinking steps for rendering
    assistantMsg._thinkingSteps = [...thinkingSteps]

    setMessages((prev) => {
      const lastIsAssistant = prev.length > 0 && prev[prev.length - 1].role === 'assistant'
      if (lastIsAssistant) return [...prev.slice(0, -1), assistantMsg]
      return [...prev, assistantMsg]
    })

    if (activeSessionId && !displayContent.startsWith('出错了')) {
      try {
        await saveMessage(activeSessionId, 'user', userMsg.content, 'text')
        await saveMessage(activeSessionId, 'assistant', displayContent, 'text', {
          intent,
          intent_confidence: intentConfidence || undefined,
          mode: resolvedMode,
          tools_used: toolsUsed,
          chunks: ragChunks.length > 0 ? ragChunks : undefined,
          quiz: quizData,
          retrieve_mode: retrieveMode || undefined,
          retrieve_strategy: retrieveStrategy || undefined,
          guardrail,
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
  }, [
    input,
    imageFile,
    imagePreview,
    loading,
    messages,
    courseId,
    onSessionCreated,
    chatMode,
    ragEnabled,
    useKb,
    isQuizMode,
    handleQuizStart,
  ])

  const handleStop = () => {
    if (quizStreaming) {
      quizCloseRef.current?.()
      setQuizStreaming(false)
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
      void (isQuizMode ? handleQuizStart() : handleSend())
    }
  }

  const isRunning = loading || quizStreaming
  const activeCapDef = CAPABILITIES.find((c) => c.value === activeCap)!
  const CapIcon = activeCapDef.icon

  return (
    <div className="flex flex-col h-full">
      {/* 顶部标题栏 */}
      <div className="border-b border-slate-200 px-6 py-3 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold text-slate-800">{courseName}</h1>
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
        className="flex-1 overflow-y-auto px-6 py-4 bg-slate-50/50"
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
      </div>

      {/* 底部输入区 */}
      <div className="border-t border-slate-200 bg-white px-4 pt-3 pb-4">
        {/* 出题配置面板（quiz 模式时展开） */}
        {isQuizMode && (
          <QuizConfigPanel value={quizConfig} onChange={setQuizConfig} />
        )}

        <div className="flex items-end gap-2">
          {/* 能力菜单 */}
          <div className="relative shrink-0" ref={capMenuRef}>
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

          {/* 工具开关 */}
          <button
            type="button"
            onClick={() => setUseKb((v) => !v)}
            title="知识库检索"
            className={`inline-flex items-center gap-1 px-2 py-2 rounded-xl border text-xs transition ${
              useKb || ragEnabled
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }`}
          >
            <FiDatabase size={14} />
            <span className="hidden sm:inline text-[11px]">知识库</span>
          </button>
          <button
            type="button"
            onClick={() => setUseWebSearch((v) => !v)}
            title="网络搜索"
            className={`inline-flex items-center gap-1 px-2 py-2 rounded-xl border text-xs transition ${
              useWebSearch
                ? 'border-indigo-400 bg-indigo-50 text-indigo-600'
                : 'border-slate-200 text-slate-400 hover:border-slate-300'
            }`}
          >
            <FiGlobe size={14} />
            <span className="hidden sm:inline text-[11px]">搜索</span>
          </button>

          {/* 图片上传（非 quiz 模式） */}
          {!isQuizMode && (
            <ImageUpload preview={imagePreview} onSelect={handleImageSelect} onClear={clearImage} />
          )}

          {/* 输入框 */}
          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isQuizMode
                  ? '输入知识点直接出题，或在上方配置面板填写详细参数…'
                  : activeCap === 'deep_solve'
                  ? '输入题目，深度推理求解…'
                  : activeCap === 'research'
                  ? '输入研究主题…'
                  : '输入问题（Shift+Enter 换行）'
              }
              rows={1}
              disabled={isRunning}
              className="w-full resize-none rounded-xl border border-slate-200 px-4 py-2.5 pr-12 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition disabled:opacity-50"
              style={{ minHeight: '42px', maxHeight: '120px' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement
                target.style.height = 'auto'
                target.style.height = Math.min(target.scrollHeight, 120) + 'px'
              }}
            />
          </div>

          {/* 发送 / 停止按钮 */}
          <button
            onClick={isRunning ? handleStop : () => void (isQuizMode ? handleQuizStart() : handleSend())}
            disabled={
              isStopping ||
              (!isRunning && !input.trim() && !imageFile && !(isQuizMode && quizConfig.topic.trim()))
            }
            className={`p-2.5 rounded-xl text-white disabled:opacity-40 disabled:cursor-not-allowed transition ${
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
