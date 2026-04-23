import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { normalizeQuizMarkdown } from '../lib/normalizeQuizMarkdown'

type Props = {
  content: string
  className?: string
}

/**
 * 与 MessageBubble 一致的 Markdown+公式渲染，用于题目解析、参考答案等长文本。
 * 对 LLM 常犯的「## 不单独成行」等做归一化，与 DeepTutor 知识页/Quiz 的预期一致。
 */
export default function FormattedMarkdown({ content, className = 'markdown-body text-sm leading-relaxed' }: Props) {
  if (!content?.trim()) return null
  const md = normalizeQuizMarkdown(content)
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {md}
      </ReactMarkdown>
    </div>
  )
}
