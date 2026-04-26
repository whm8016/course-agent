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

export interface GuardrailInfo {
  safe: boolean
  risk_type: string
  risk_score: number
  tip: string
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
  guardrail?: GuardrailInfo
  hallucination?: HallucinationInfo
}

export interface SSEEvent {
  type: 'thinking' | 'thinking_chunk' | 'tool_call' | 'tool_result' | 'answer' | 'quiz' | 'done' | 'error' | 'token'
  content?: string
  tool?: string
  input?: Record<string, unknown>
  chunks?: RagChunk[]
  quiz?: QuizData
  metadata?: AgentMetadata
  stage?: string
  call_state?: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  image?: string
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
    guardrail?: GuardrailInfo
    hallucination?: HallucinationInfo
    stopped?: boolean
    stage?: string
    call_state?: string
    timestamp?: number
  }
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
  is_admin?: boolean
  summary_memory?: string
  profile_memory?: string
}

export interface AuthResponse {
  token: string
  user: User
}
