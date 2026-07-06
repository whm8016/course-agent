/**
 * Toast 全局事件单例（无状态库的最轻方案）。
 *
 * 模块级数组 + 订阅者：<ToastViewport/>（在 components/ui.tsx）挂一次，
 * 任何地方 toast.success/error/info(...) 触发。替 App.tsx 手写 toast。
 *
 * 放在 lib/toast.ts（纯 .ts）而非 ui.tsx：toast 是「服务」不是「视觉组件」，
 * 拆出来让 ui.tsx 只导出组件（满足 react-refresh/only-export-components），
 * 调用方从 lib/toast import 也比从 UI 组件库 import 语义更清晰。
 */
import { useEffect, useState } from 'react'

export type ToastVariant = 'success' | 'error' | 'info'

export interface ToastItem {
  id: number
  variant: ToastVariant
  message: string
}

let _toasts: ToastItem[] = []
type Listener = (items: ToastItem[]) => void
let _listeners: Listener[] = []
let _nextId = 1

function emit() {
  const snapshot = [..._toasts]
  _listeners.forEach((l) => l(snapshot))
}

function show(variant: ToastVariant, message: string, duration = 3000) {
  const id = _nextId++
  _toasts = [..._toasts, { id, variant, message }]
  emit()
  setTimeout(() => dismiss(id), duration)
}

function dismiss(id: number) {
  _toasts = _toasts.filter((t) => t.id !== id)
  emit()
}

export const toast = {
  success: (m: string) => show('success', m),
  error: (m: string) => show('error', m, 5000),
  info: (m: string) => show('info', m),
  dismiss,
}

/** 订阅 toast 列表变化，供 ToastViewport 使用。 */
export function useToasts(): ToastItem[] {
  const [items, setItems] = useState<ToastItem[]>(_toasts)
  useEffect(() => {
    const l: Listener = (x) => setItems(x)
    _listeners.push(l)
    return () => {
      _listeners = _listeners.filter((x) => x !== l)
    }
  }, [])
  return items
}
