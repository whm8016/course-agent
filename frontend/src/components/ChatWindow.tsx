import { useState, useRef, useEffect, useCallback } from 'react'
import { FiSend, FiSquare } from 'react-icons/fi'
import MessageBubble from './MessageBubble'
import ImageUpload from './ImageUpload'
import { chatStream, uploadImage, fetchMessages, saveMessage, createSession, updateSessionMode } from '../services/api'
import type { Message, Session, SSEEvent, RagChunk, QuizData, ChatMode } from '../types'

interface Props {
  courseId: string
  courseName: string
  sessionId: string | null
  sessionMode?: ChatMode
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

const MODE_OPTIONS: Array<{ value: ChatMode; label: string }> = [
  { value: 'chat', label: '通用问答' },
  { value: 'deep_solve', label: '深度解题' },
  { value: 'quiz', label: '测验出题' },
  { value: 'research', label: '深度研究' },
  { value: 'vision', label: '图像分析' },
]

export default function ChatWindow({ courseId, courseName, sessionId, sessionMode, onSessionCreated }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingStarted, setStreamingStarted] = useState(false)
  const [streamPhase, setStreamPhase] = useState<'retrieving' | 'generating' | null>(null)
  const [isStopping, setIsStopping] = useState(false)
  const [chatMode, setChatMode] = useState<ChatMode>('chat')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const currentSessionRef = useRef<string | null>(sessionId)
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    currentSessionRef.current = sessionId
  }, [sessionId])

  useEffect(() => {
    setChatMode(sessionMode || 'chat')
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: loading ? 'auto' : 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
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

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text && !imageFile) return
    if (loading) return

    let activeSessionId = currentSessionRef.current

    if (!activeSessionId) {
      try {
        const title = (text || '图片分析').slice(0, 20) || '新对话'
        const session = await createSession(courseId, title, chatMode)
        activeSessionId = session.id
        currentSessionRef.current = session.id
        onSessionCreated(session)
      } catch {
        // fall through — chat will still work, just not persisted
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

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    clearImage()
    setLoading(true)
    setIsStopping(false)
    setStreamingStarted(false)
    setStreamPhase('retrieving')
    const controller = new AbortController()
    abortControllerRef.current = controller

    const history = messages.map((m) => ({ role: m.role, content: m.content }))

    const thinkingSteps: Message[] = []
    let answerContent = ''
    let ragChunks: RagChunk[] = []
    let quizData: QuizData | undefined
    let intent = ''
    let resolvedMode: ChatMode = chatMode
    let toolsUsed: string[] = []
    let retrieveMode = ''
    let retrieveStrategy = ''

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
            setStreamPhase('retrieving')
            thinkingSteps.push({
              role: 'assistant',
              content: event.content || '',
              type: 'thinking',
            })
            break

          case 'tool_call':
            setStreamPhase('retrieving')
            thinkingSteps.push({
              role: 'assistant',
              content: '',
              type: 'tool_call',
              metadata: { tool: event.tool, toolInput: event.input as Record<string, unknown> },
            })
            break

          case 'tool_result':
            setStreamPhase('retrieving')
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
            setStreamPhase('generating')
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
            setStreamPhase('generating')
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
            resolvedMode = (event.metadata?.mode as ChatMode) || chatMode
            toolsUsed = event.metadata?.tools_used || []
            retrieveMode = event.metadata?.retrieve_mode || ''
            retrieveStrategy = event.metadata?.retrieve_strategy || ''
            break

          case 'error':
            answerContent = `出错了: ${event.content}`
            break
        }
      },
      (err) => {
        answerContent = `出错了: ${err}`
      },
    )
    abortControllerRef.current = null

    if (streamResult.aborted) {
      setLoading(false)
      setIsStopping(false)
      setStreamingStarted(false)
      setStreamPhase(null)
      return
    }

    const assistantMsg: Message = {
      role: 'assistant',
      content: answerContent,
      metadata: {
        intent,
        mode: resolvedMode,
        chunks: ragChunks.length > 0 ? ragChunks : undefined,
        quiz: quizData,
        tools_used: toolsUsed.length > 0 ? toolsUsed : undefined,
        retrieve_mode: retrieveMode || undefined,
        retrieve_strategy: retrieveStrategy || undefined,
      },
    }
    // @ts-expect-error attach thinking steps for rendering
    assistantMsg._thinkingSteps = [...thinkingSteps]

    setMessages((prev) => {
      const lastIsAssistant = prev.length > 0 && prev[prev.length - 1].role === 'assistant'
      if (lastIsAssistant) {
        return [...prev.slice(0, -1), assistantMsg]
      }
      return [...prev, assistantMsg]
    })

    if (activeSessionId && !answerContent.startsWith('出错了')) {
      try {
        await saveMessage(activeSessionId, 'user', userMsg.content, 'text')
        await saveMessage(activeSessionId, 'assistant', answerContent, 'text', {
          intent,
          mode: resolvedMode,
          tools_used: toolsUsed,
          chunks: ragChunks.length > 0 ? ragChunks : undefined,
          quiz: quizData,
          retrieve_mode: retrieveMode || undefined,
          retrieve_strategy: retrieveStrategy || undefined,
        })
      } catch {
        /* persistence is best-effort */
      }
    }

    setLoading(false)
    setIsStopping(false)
    setStreamingStarted(false)
    setStreamPhase(null)
  }, [input, imageFile, imagePreview, loading, messages, courseId, onSessionCreated, chatMode])

  const handleStop = () => {
    if (!loading) return
    setIsStopping(true)
    abortControllerRef.current?.abort()
  }

  const handleModeChange = async (nextMode: ChatMode) => {
    setChatMode(nextMode)
    if (currentSessionRef.current) {
      try {
        await updateSessionMode(currentSessionRef.current, nextMode)
      } catch {
        // keep local mode even if update fails
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-slate-200 px-6 py-4 bg-white/80 backdrop-blur-sm">
        <h1 className="text-lg font-semibold text-slate-800">{courseName} - 学习助手</h1>
        <div className="flex items-center justify-between mt-1 gap-3">
          <p className="text-xs text-slate-400">多 Agent 编排 · RAG 知识检索 · 智能出题</p>
          <select
            value={chatMode}
            onChange={(e) => void handleModeChange(e.target.value as ChatMode)}
            className="text-xs border border-slate-200 rounded-md px-2 py-1 bg-white text-slate-600"
          >
            {MODE_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 bg-slate-50/50">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <div className="text-5xl mb-4">💬</div>
            <p className="text-lg font-medium">开始提问吧</p>
            <p className="text-sm mt-1">输入问题、要求出题或上传图片</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            thinkingSteps={(msg as unknown as Record<string, unknown>)._thinkingSteps as Message[] | undefined}
          />
        ))}
        {loading && !streamingStarted && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" />
              </div>
              <p className="text-xs text-slate-500 mt-2">
                {streamPhase === 'generating' ? '正在生成回答...' : '正在检索资料...'}
              </p>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-200 bg-white px-6 py-4">
        <div className="flex items-end gap-3">
          <ImageUpload preview={imagePreview} onSelect={handleImageSelect} onClear={clearImage} />
          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题 · 要求出题 · 请求总结（Shift+Enter 换行）"
              rows={1}
              className="w-full resize-none rounded-xl border border-slate-200 px-4 py-3 pr-12 text-sm focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition"
              style={{ minHeight: '44px', maxHeight: '120px' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement
                target.style.height = 'auto'
                target.style.height = Math.min(target.scrollHeight, 120) + 'px'
              }}
            />
          </div>
          <button
            onClick={loading ? handleStop : handleSend}
            disabled={isStopping || (!loading && !input.trim() && !imageFile)}
            className={`p-3 rounded-xl text-white disabled:opacity-40 disabled:cursor-not-allowed transition ${
              loading ? 'bg-rose-600 hover:bg-rose-700' : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
            title={loading ? '停止生成' : '发送'}
          >
            {loading ? <FiSquare size={18} /> : <FiSend size={18} />}
          </button>
        </div>
      </div>
    </div>
  )
}
