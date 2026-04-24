import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Message } from '../types'
import ThinkingProcess from './ThinkingProcess'
import SourceCard from './SourceCard'
import QuizCard from './QuizCard'

interface Props {
  message: Message
  thinkingSteps?: Message[]
  courseId?: string
}

function looksLikeMath(expr: string): boolean {
  const t = expr.trim()
  if (!t) return false
  if (/\\[a-zA-Z]+/.test(t)) return true
  if (/[_^]/.test(t)) return true
  if (/=/.test(t) && /[a-zA-Z]/.test(t)) return true
  if (/^[a-zA-Z](?:_\{?[^}]+\}?)?$/.test(t)) return true
  return false
}

function normalizeMathDelimiters(content: string): string {
  if (!content) return content

  let normalized = content
    .replace(/\\\[((?:.|\n)+?)\\\]/g, (_, expr: string) => `$$${expr.trim()}$$`)
    .replace(/\\\(((?:.|\n)+?)\\\)/g, (_, expr: string) => `$${expr.trim()}$`)

  normalized = normalized.replace(
    /\[\s+([^\[\]\n]{1,500}?)\s+\]/g,
    (match, inner: string) => (looksLikeMath(inner) ? `$$${inner.trim()}$$` : match),
  )

  normalized = normalized.replace(
    /(^|[^$])\(\s+([^()\n]{1,200}?)\s+\)/g,
    (match, prefix: string, inner: string) =>
      looksLikeMath(inner) ? `${prefix}$${inner.trim()}$` : match,
  )

  return normalized
}

export default function MessageBubble({ message, thinkingSteps, courseId }: Props) {
  const isUser = message.role === 'user'
  const renderedContent = normalizeMathDelimiters(message.content || '')

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%] rounded-2xl px-4 py-3 bg-indigo-600 text-white rounded-br-md">
          {message.image && (
            <img src={message.image} alt="上传的图片" className="max-w-[280px] rounded-lg mb-2" />
          )}
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm">
        {thinkingSteps && thinkingSteps.length > 0 && (
          <ThinkingProcess steps={thinkingSteps} />
        )}

        {message.content && (
          <div className="markdown-body text-sm leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
              {renderedContent}
            </ReactMarkdown>
          </div>
        )}

        {message.metadata?.chunks && message.metadata.chunks.length > 0 && (
          <SourceCard chunks={message.metadata.chunks} />
        )}

        {message.metadata?.quiz && (
          <QuizCard quiz={message.metadata.quiz} courseId={courseId} />
        )}

        {message.metadata?.guardrail && !message.metadata.guardrail.safe && (
          <div className="mt-2 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-700">
            <span className="font-medium">安全提示：</span>{message.metadata.guardrail.tip}
          </div>
        )}

        {message.metadata?.hallucination && message.metadata.hallucination.tip && (
          <div className="mt-2 px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-xs text-blue-700">
            <span className="font-medium">可信度：</span>
            {message.metadata.hallucination.tip}
            {message.metadata.hallucination.confidence > 0 && (
              <span className="ml-1 text-blue-500">
                ({Math.round(message.metadata.hallucination.confidence * 100)}%)
              </span>
            )}
          </div>
        )}

        {message.metadata?.intent && (
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
            {message.metadata.intent === 'chitchat' && <span>💬 闲聊</span>}
            {message.metadata.intent === 'knowledge' && <span>📖 知识问答</span>}
            {message.metadata.intent === 'teach' && <span>📖 知识问答</span>}
            {message.metadata.intent === 'quiz' && <span>📝 测验模式</span>}
            {message.metadata.intent === 'summarize' && <span>📋 学习总结</span>}
            {message.metadata.intent === 'vision' && <span>🔍 图像分析</span>}
            {message.metadata.mode === 'deep_solve' && <span>🧠 深度解题</span>}
            {message.metadata.mode === 'research' && <span>🔎 深度研究</span>}
            {message.metadata.tools_used && message.metadata.tools_used.length > 0 && (
              <span className="text-slate-300">
                · 使用了 {message.metadata.tools_used.join(', ')}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
