export interface Course {
  id: string
  name: string
  icon: string
  description: string
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
  type: 'thinking' | 'tool_call' | 'tool_result' | 'answer' | 'quiz' | 'done' | 'error' | 'token'
  content?: string
  tool?: string
  input?: Record<string, unknown>
  chunks?: RagChunk[]
  quiz?: QuizData
  metadata?: AgentMetadata
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
  summary_memory?: string
  profile_memory?: {
    level?: string
    style?: string
    goal?: string
    preferred_mode?: string
  }
}

export interface AuthResponse {
  token: string
  user: User
}
