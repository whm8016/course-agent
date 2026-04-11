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
}

export default function MessageBubble({ message, thinkingSteps }: Props) {
  const isUser = message.role === 'user'

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
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {message.metadata?.chunks && message.metadata.chunks.length > 0 && (
          <SourceCard chunks={message.metadata.chunks} />
        )}

        {message.metadata?.quiz && (
          <QuizCard quiz={message.metadata.quiz} />
        )}

        {message.metadata?.intent && (
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
            {message.metadata.intent === 'teach' && <span>📖 知识问答</span>}
            {message.metadata.intent === 'quiz' && <span>📝 测验模式</span>}
            {message.metadata.intent === 'summarize' && <span>📋 学习总结</span>}
            {message.metadata.intent === 'vision' && <span>🔍 图像分析</span>}
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
