import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  MessageCircle,
  BookOpen,
  ClipboardList,
  AlignLeft,
  ScanEye,
  BrainCircuit,
  Search,
  Image as ImageIcon,
  Paperclip,
  type LucideIcon,
} from 'lucide-react'
import type { Message } from '../../types'
import ThinkingProcess from './ThinkingProcess'
import SourceCard from './SourceCard'
import QuizCard from '../quiz/QuizCard'

interface Props {
  message: Message
  thinkingSteps?: Message[]
  courseId?: string
  isStreaming?: boolean
}


/**
 * 表格行内的公式含 | 会被 GFM 当作列分隔符，导致列错位 / 公式截断。
 * 将公式内的 | 替换为 \vert（KaTeX 等价），\t 替换为空格。
 */
function escapeTablePipes(content: string): string {
  return content.replace(/^(\|.+)$/gm, (line: string) => {
    line = line.replace(/\t/g, ' ')

    const segments: string[] = []
    let i = 0
    while (i < line.length) {
      // $$ 块公式
      if (line[i] === '$' && line[i + 1] === '$') {
        const end = line.indexOf('$$', i + 2)
        if (end !== -1) {
          segments.push(line.slice(i, end + 2).replace(/\|/g, '\\vert '))
          i = end + 2
          continue
        }
      }
      // $ 行内公式
      if (line[i] === '$') {
        const end = line.indexOf('$', i + 1)
        if (end !== -1) {
          segments.push(line.slice(i, end + 1).replace(/\|/g, '\\vert '))
          i = end + 1
          continue
        }
      }
      // \( ... \) 行内公式
      if (line[i] === '\\' && line[i + 1] === '(') {
        const end = line.indexOf('\\)', i + 2)
        if (end !== -1) {
          segments.push(line.slice(i, end + 2).replace(/\|/g, '\\vert '))
          i = end + 2
          continue
        }
      }
      segments.push(line[i])
      i++
    }
    return segments.join('')
  })
}

function normalizeMathDelimiters(content: string): string {
  if (!content) return content

  return escapeTablePipes(content)
    // 表格行（含 | 的行）内的 <br> → 空格；其余 <br> → 真换行
    .replace(/^(.*\|.*)<br\s*\/?>(.*\|.*)$/gim, '$1 $2')
    .replace(/<br\s*\/?>/gi, '\n')
    // \[...\] → 独立成行的 $$...$$
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, expr: string) => `\n$$\n${expr.trim()}\n$$\n`)
    // \(...\) → 行内 $...$，允许换行（多行行内公式）
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, expr: string) => `$${expr.trim()}$`)
    // 已经是 $$...$$ 的块公式，保证前后各有一个空行，防止被 GFM 当普通文本解析
    .replace(/([^\n])\$\$((?:(?!\n\n)[\s\S])+?)\$\$([^\n])/g, (_, pre, expr: string, post) =>
      `${pre}\n$$\n${expr.trim()}\n$$\n${post}`,
    )
}

export default function MessageBubble({ message, thinkingSteps, courseId, isStreaming }: Props) {
  const isUser = message.role === 'user'
  const renderedContent = normalizeMathDelimiters(message.content || '')

  if (isUser) {
    const userContent = renderedContent
    const docAtts = (message.attachments || []).filter((a) => a.type !== 'image')
    // 无 blob 预览（历史回显）时图片以文件名 chip 呈现：/api/uploads 需鉴权，不能直接 <img>
    const imageChips = message.image ? [] : (message.attachments || []).filter((a) => a.type === 'image')
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[92%] md:max-w-[75%] rounded-[var(--radius-lg)] px-4 py-3 bg-ink text-white rounded-br-[3px]">
          {message.image && (
            <img src={message.image} alt="上传的图片" className="max-w-[280px] rounded-[var(--radius)] mb-2" />
          )}
          {(imageChips.length > 0 || docAtts.length > 0) && (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {imageChips.map((a, i) => (
                <span
                  key={`img-${i}`}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-[var(--radius-sm)] bg-white/15 text-xs"
                >
                  <ImageIcon size={12} strokeWidth={1.5} />
                  {a.filename || '图片'}
                </span>
              ))}
              {docAtts.map((a, i) => (
                <span
                  key={`doc-${i}`}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-[var(--radius-sm)] bg-white/15 text-xs"
                >
                  <Paperclip size={12} strokeWidth={1.5} />
                  {a.filename || '文档'}
                </span>
              ))}
            </div>
          )}
          <div className="text-sm leading-relaxed markdown-body markdown-user">
            <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]}>
              {userContent}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    )
  }

  // intent / mode → 图标 badge（原 emoji 版的等价重构：条件与文案不变，只换呈现）
  const meta = message.metadata
  const intentBadges: { icon: LucideIcon; label: string }[] = []
  if (meta) {
    if (meta.intent === 'chitchat') intentBadges.push({ icon: MessageCircle, label: '闲聊' })
    else if (meta.intent === 'knowledge' || meta.intent === 'teach')
      intentBadges.push({ icon: BookOpen, label: '知识问答' })
    else if (meta.intent === 'quiz') intentBadges.push({ icon: ClipboardList, label: '测验模式' })
    else if (meta.intent === 'summarize') intentBadges.push({ icon: AlignLeft, label: '学习总结' })
    else if (meta.intent === 'vision') intentBadges.push({ icon: ScanEye, label: '图像分析' })
    if (meta.mode === 'deep_solve') intentBadges.push({ icon: BrainCircuit, label: '深度解题' })
    if (meta.mode === 'research') intentBadges.push({ icon: Search, label: '深度研究' })
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[95%] md:max-w-[80%] rounded-[var(--radius-lg)] px-4 py-3 bg-surface border border-line text-ink rounded-bl-[3px]">
        {thinkingSteps && thinkingSteps.length > 0 && (
          <ThinkingProcess steps={thinkingSteps} isStreaming={isStreaming} />
        )}

        {message.content && (
          <div className="markdown-body text-sm leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]}>
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

        {message.metadata?.hallucination && message.metadata.hallucination.tip && (
          <div className="mt-2 px-3 py-1.5 rounded-[var(--radius)] bg-info-bg border border-line text-xs text-info-fg">
            <span className="font-medium">可信度：</span>
            {message.metadata.hallucination.tip}
            {message.metadata.hallucination.confidence > 0 && (
              <span className="ml-1">({Math.round(message.metadata.hallucination.confidence * 100)}%)</span>
            )}
          </div>
        )}

        {message.metadata?.intent && (
          <div className="mt-2 flex items-center gap-2 flex-wrap text-xs text-muted">
            {intentBadges.map((b, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                <b.icon size={12} strokeWidth={1.5} />
                {b.label}
              </span>
            ))}
            {message.metadata.tools_used && message.metadata.tools_used.length > 0 && (
              <span className="text-muted/70">· 使用了 {message.metadata.tools_used.join(', ')}</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
