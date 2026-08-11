export type KBStatus = 'pending' | 'indexing' | 'ready' | 'error' | 'paused'

export interface Course {
  id: string
  name: string
  icon: string
  description: string
  /** 后端附带的知识库状态，null 表示该课程没有 KB（仅有内置 system prompt） */
  kb_status?: KBStatus | null
  /** kb_status === 'ready' 时为 true，前端用它决定走 /api/chat 还是 /api/chat/lightrag */
  rag_enabled?: boolean
  /** 该课程已就绪的索引后端（'lightrag' | 'llamaindex_pg'）。学生端问答用它决定
   *  检索模式选择器是否显示（仅 lightrag 就绪时显示 mix/naive/local；auto 永远可用）。 */
  index_backends?: string[]
  /** 'builtin' | 'kb'，仅做来源标识 */
  source?: 'builtin' | 'kb'
}

export interface RagChunk {
  content: string
  source: string
  score: number
}

export interface QuizQuestion {
  question: string
  options: string[]
  answer: string
  explanation: string
}

export interface QuizData {
  questions: QuizQuestion[]
}

export interface HallucinationInfo {
  grounded: boolean
  confidence: number
  tip: string
}

export interface AgentMetadata {
  intent?: string
  intent_confidence?: number
  mode?: string
  tools_used?: string[]
  retrieve_mode?: string
  retrieve_strategy?: string
  hallucination?: HallucinationInfo
  usage?: MessageUsage
}

export interface SSEEvent {
  type: 'thinking' | 'thinking_chunk' | 'tool_call' | 'tool_result' | 'answer' | 'quiz' | 'done' | 'error' | 'token' | 'skill_output' | 'turn_started'
  content?: string
  /** error 事件携带的错误文案。后端 stream_bus 发的是 `message` 字段（非 content），
   *  前端读 error 事件时须同时认 content / message，否则显示「出错了: undefined」。 */
  message?: string
  tool?: string
  input?: Record<string, unknown>
  chunks?: RagChunk[]
  quiz?: QuizData
  metadata?: AgentMetadata
  stage?: string
  call_state?: string
  /** turn_started 事件携带：本回合 turn id，供"立即回答"按钮 POST /chat/answer_now 用 */
  turn_id?: string
}

export interface AttachmentInfo {
  type: 'image' | 'file' | 'pdf'
  url: string
  filename?: string
  mime_type?: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  image?: string
  attachments?: AttachmentInfo[]
  type?: 'text' | 'thinking' | 'tool_call' | 'tool_result' | 'quiz'
  metadata?: {
    intent?: string
    intent_confidence?: number
    tool?: string
    toolInput?: Record<string, unknown>
    chunks?: RagChunk[]
    quiz?: QuizData
    tools_used?: string[]
    mode?: string
    retrieve_mode?: string
    retrieve_strategy?: string
    hallucination?: HallucinationInfo
    stopped?: boolean
    stage?: string
    call_state?: string
    usage?: MessageUsage
    timestamp?: number
  }
}

// 本轮 token 用量（done 事件 metadata.usage 透传，学生端气泡底部小字）。
// cost_usd 仅 expose_cost_to_student 开启时存在；token 数始终可见。
export interface MessageUsage {
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  cost_usd?: number
}

export interface Session {
  id: string
  course_id: string
  title: string
  mode?: ChatMode
  created_at: number
  updated_at: number
}

export type ChatMode = 'chat' | 'deep_solve' | 'quiz' | 'research' | 'vision' | 'summarize'

export interface ChatSession {
  id: string
  courseId: string
  title: string
  messages: Message[]
  createdAt: number
}

export interface User {
  id: string
  username: string
  display_name: string
  role?: 'student' | 'teacher' | 'admin'
  is_admin?: boolean
}

export interface AuthResponse {
  token: string
  user: User
}
