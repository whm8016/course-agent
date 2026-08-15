/**
 * MessageBubble 流式 Markdown 渲染单测。
 * 背景：曾为避免每 token 全量重跑 remark/rehype/KaTeX，流式中渲纯文本（**、$$ 原样显示），
 * done 后才换 Markdown，观感割裂。改为节流快照后：流式中即渲 Markdown，重解析频率封顶。
 * 覆盖三点：流式中渲 Markdown / 150ms 节流生效 / done 立即渲最终全文。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, act } from '@testing-library/react'
import MessageBubble from '../components/chat/MessageBubble'
import type { Message } from '../types'

const mk = (content: string): Message => ({ role: 'assistant', content })

describe('MessageBubble 流式 Markdown 渲染', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('流式中即渲染 Markdown，不再显示 ** 原始符号', () => {
    const { container } = render(<MessageBubble message={mk('**加粗内容**')} isStreaming />)
    expect(container.querySelector('strong')?.textContent).toBe('加粗内容')
    expect(container.textContent).not.toContain('**')
  })

  it('流式中即渲染行内公式（KaTeX）', () => {
    const { container } = render(<MessageBubble message={mk('电压 $U=IR$ 很重要')} isStreaming />)
    expect(container.querySelector('.katex')).not.toBeNull()
  })

  it('节流：150ms 内多次 token 只渲首个快照，到点后追上最新', () => {
    const view = render(<MessageBubble message={mk('第一段')} isStreaming />)
    expect(view.container.textContent).toContain('第一段')

    // 推进 10ms 后送入新 token：距上次渲染 <150ms，仍是旧快照（无 strong）
    act(() => {
      vi.advanceTimersByTime(10)
    })
    view.rerender(<MessageBubble message={mk('第一段\n\n**第二段**')} isStreaming />)
    expect(view.container.querySelector('strong')).toBeNull()

    // 到点后定时器触发，快照追上最新内容
    act(() => {
      vi.advanceTimersByTime(150)
    })
    expect(view.container.querySelector('strong')?.textContent).toBe('第二段')
  })

  it('流结束立即渲染最终全文（不等下一个节流周期）', () => {
    const view = render(<MessageBubble message={mk('开头')} isStreaming />)
    act(() => {
      vi.advanceTimersByTime(10)
    })
    view.rerender(<MessageBubble message={mk('**最终答案**')} isStreaming={false} />)
    expect(view.container.querySelector('strong')?.textContent).toBe('最终答案')
  })

  it('非流式历史消息渲染 Markdown 不受影响', () => {
    const { container } = render(<MessageBubble message={mk('## 标题\n\n正文')} />)
    expect(container.querySelector('h2')?.textContent).toBe('标题')
  })
})
