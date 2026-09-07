/**
 * 流式贴底跟随 hook（ChatWindow 主消息流 / QuizCard 追问面板共用）。
 *
 * 贴底（距底 <40px）期间新内容到达即跟随；用户上拉脱离即停，拉回底部恢复。
 * 跟随只直接赋值本容器 scrollTop（瞬时、只动本容器）——绝不用 bottomRef.scrollIntoView：
 * 后者会连带滚动祖先链（整个页面被反复压到底），且 smooth 在高频流式下排队动画、
 * 与用户手动滚动打架，是"拉不动/定不住"的主要来源（主消息流与追问面板各踩过一次）。
 * 阈值 40：流式时用户想看的内容多在底部附近，80px 内一旦上拉就被瞬时拽回、定不住
 * （调参史 80→40；两处面板必须同口径，故收口到此 hook 单点维护）。
 */
import { useCallback, useRef, useState } from 'react'

export function useStickToBottom() {
  const ref = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)

  const onScroll = useCallback(() => {
    const el = ref.current
    if (!el) return
    setIsAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 40)
  }, [])

  /** 滚到底并恢复跟随（发送新消息 / "回到底部"按钮 / 流式跟随 effect）。
   *  跟随 effect 形态以 if (isAtBottom) 守卫调用——只想要"滚"的场合必须自带守卫，
   *  否则 setIsAtBottom(true) 会把已脱离贴底的容器意外拉回跟随。 */
  const scrollToBottom = useCallback(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
    setIsAtBottom(true)
  }, [])

  return { ref, isAtBottom, onScroll, scrollToBottom }
}
