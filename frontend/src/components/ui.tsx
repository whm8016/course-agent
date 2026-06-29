import { type ReactNode } from 'react'

/** 共享 UI 组件（现有 slate/indigo 风格，供新管理页面复用）。 */

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  overlayCloses = true,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
  /** 点遮罩是否关闭（编辑表单类应设 false，防误关丢输入）。默认 true。 */
  overlayCloses?: boolean
}) {
  if (!open) return null
  return (
    <div
      className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
      onClick={overlayCloses ? onClose : undefined}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-800">{title}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">✕</button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
        {footer && (
          <div className="px-5 py-3 border-t border-slate-100 flex justify-end gap-2">{footer}</div>
        )}
      </div>
    </div>
  )
}

export function Badge({
  children,
  color = 'slate',
}: {
  children: ReactNode
  color?: 'slate' | 'green' | 'amber' | 'red' | 'indigo' | 'blue'
}) {
  const colors: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-600',
    green: 'bg-green-100 text-green-700',
    amber: 'bg-amber-100 text-amber-700',
    red: 'bg-red-100 text-red-700',
    indigo: 'bg-indigo-100 text-indigo-700',
    blue: 'bg-blue-100 text-blue-700',
  }
  return (
    <span className={`inline-block text-xs px-2 py-0.5 rounded-full ${colors[color]}`}>{children}</span>
  )
}

export function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative w-9 h-5 rounded-full transition shrink-0 ${
        checked ? 'bg-indigo-600' : 'bg-slate-300'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition ${
          checked ? 'translate-x-4' : ''
        }`}
      />
    </button>
  )
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`bg-white rounded-xl border border-slate-200 shadow-sm ${className}`}>{children}</div>
  )
}

export function EmptyState({
  icon = '📭',
  title,
  hint,
}: {
  icon?: string
  title: string
  hint?: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-400 gap-2">
      <span className="text-4xl">{icon}</span>
      <p className="text-sm font-medium">{title}</p>
      {hint && <p className="text-xs">{hint}</p>}
    </div>
  )
}

export function StatusDot({ status }: { status: 'connected' | 'error' | 'connecting' | 'disabled' | string }) {
  const colors: Record<string, string> = {
    connected: 'bg-green-500',
    error: 'bg-red-500',
    connecting: 'bg-amber-500',
    disabled: 'bg-slate-300',
  }
  return <span className={`inline-block w-2 h-2 rounded-full ${colors[status] || 'bg-slate-300'}`} />
}
