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

export interface AgentMetadata {
  intent?: string
  tools_used?: string[]
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
    tool?: string
    toolInput?: Record<string, unknown>
    chunks?: RagChunk[]
    quiz?: QuizData
    tools_used?: string[]
  }
}

export interface Session {
  id: string
  course_id: string
  title: string
  created_at: number
  updated_at: number
}

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
}

export interface AuthResponse {
  token: string
  user: User
}
