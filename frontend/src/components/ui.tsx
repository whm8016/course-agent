/**
 * 共享 UI 组件库 —— 暖白极简编辑风（教育类 SaaS）
 *
 * 设计令牌来自 src/index.css 的 @theme / :root（见 minimalist-ui skill）：
 *   颜色 bg-canvas/surface/surface-2、text-ink/ink-soft/muted、border-line、accent
 *   语义 muted-pastel：ok/info/warn/danger/neutral（各 -bg/-fg）
 *   圆角用 rounded-[var(--radius)] / -sm / -lg；阴影用 shadow-[var(--shadow-hover)]
 *
 * 迁移策略：现有 6 个组件（Modal/Badge/Toggle/Card/EmptyState/StatusDot）
 * 导出名 + 核心 props 接口保持不变，只换内部样式 + 加可选 prop，让 BotPage/
 * LlmProviderPage/NotebookPage/McpSettingsPage/SkillKnowledgePage 这 5 个
 * 现有 import 方无感升级。新组件用 token 类，供 Phase C 逐页迁移复用。
 */
import {
  type ReactNode,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type TextareaHTMLAttributes,
  type SelectHTMLAttributes,
} from 'react'
import { createPortal } from 'react-dom'
import { useToasts, toast } from '../lib/toast'
import type { ToastVariant } from '../lib/toast'
import {
  X,
  Inbox,
  Loader2,
  ChevronDown,
  Check,
  AlertTriangle,
  Info,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/* ─── 基础：lucide 图标包装，统一细描边（strokeWidth 1.5）─────────────────── */
export function Icon({
  icon: I,
  size = 16,
  strokeWidth = 1.5,
  className,
}: {
  icon: LucideIcon
  size?: number
  strokeWidth?: number
  className?: string
}) {
  return <I size={size} strokeWidth={strokeWidth} className={className} />
}

/* ─── Card ───────────────────────────────────────────────────────────── */
export function Card({
  children,
  className = '',
  hover = false,
}: {
  children: ReactNode
  className?: string
  /** 可交互卡片开 hover 极淡阴影（默认关，分层靠细线 + 留白）*/
  hover?: boolean
}) {
  return (
    <div
      className={`bg-surface border border-line rounded-[var(--radius)] ${
        hover ? 'transition-shadow hover:shadow-[var(--shadow-hover)]' : ''
      } ${className}`}
    >
      {children}
    </div>
  )
}

/* ─── Modal ──────────────────────────────────────────────────────────── */
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
      className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center p-4"
      onClick={overlayCloses ? onClose : undefined}
    >
      <div
        className="bg-surface rounded-[var(--radius-lg)] border border-line w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-line">
          <h2 className="font-serif text-base text-ink tracking-tight">{title}</h2>
          <button
            onClick={onClose}
            className="text-muted hover:text-ink transition p-1 -mr-1 rounded-[var(--radius-sm)] hover:bg-surface-2"
            aria-label="关闭"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
        {footer && (
          <div className="px-5 py-3 border-t border-line flex justify-end gap-2">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── Badge ──────────────────────────────────────────────────────────── */
// 5 个语义色（muted pastel）；旧色名做别名兼容，5 个现有页面无需改。
type BadgeColor = 'neutral' | 'ok' | 'info' | 'warn' | 'danger'
const BADGE_ALIAS: Record<string, BadgeColor> = {
  neutral: 'neutral', ok: 'ok', info: 'info', warn: 'warn', danger: 'danger',
  // 旧名 → 新名
  slate: 'neutral', gray: 'neutral', grey: 'neutral',
  green: 'ok',
  blue: 'info', indigo: 'info',
  amber: 'warn', yellow: 'warn', orange: 'warn',
  red: 'danger', rose: 'danger',
}
const BADGE_STYLES: Record<BadgeColor, string> = {
  neutral: 'bg-neutral-bg text-neutral-fg',
  ok: 'bg-ok-bg text-ok-fg',
  info: 'bg-info-bg text-info-fg',
  warn: 'bg-warn-bg text-warn-fg',
  danger: 'bg-danger-bg text-danger-fg',
}
export function Badge({
  children,
  color = 'neutral',
}: {
  children: ReactNode
  /** 新名 neutral/ok/info/warn/danger；旧名 slate/green/indigo/blue/amber/red 自动别名 */
  color?: string
}) {
  const c = BADGE_ALIAS[color] ?? 'neutral'
  return (
    <span
      className={`inline-block text-[10px] font-medium uppercase tracking-[0.05em] leading-none px-2 py-1 rounded-full ${BADGE_STYLES[c]}`}
    >
      {children}
    </span>
  )
}

/* ─── Toggle ─────────────────────────────────────────────────────────── */
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
      className={`relative w-9 h-5 rounded-full transition-colors shrink-0 ${
        checked ? 'bg-ink' : 'bg-line'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      aria-pressed={checked}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-4 h-4 bg-surface rounded-full transition-transform ${
          checked ? 'translate-x-4' : ''
        }`}
      />
    </button>
  )
}

/* ─── EmptyState（icon 兼容 emoji string 与 LucideIcon）────────────────── */
export function EmptyState({
  icon,
  title,
  hint,
}: {
  /** LucideIcon 组件（新代码推荐）或 emoji 字符串（旧代码兼容，如 '🤖'）*/
  icon?: LucideIcon | string
  title: string
  hint?: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-muted gap-2.5">
      {icon === undefined ? (
        <Inbox size={32} strokeWidth={1.5} className="text-muted" />
      ) : typeof icon === 'string' ? (
        <span className="text-3xl leading-none">{icon}</span>
      ) : (
        <Icon icon={icon} size={32} className="text-muted" />
      )}
      <p className="font-serif text-sm text-ink">{title}</p>
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  )
}

/* ─── StatusDot ──────────────────────────────────────────────────────── */
export function StatusDot({
  status,
}: {
  status: 'connected' | 'error' | 'connecting' | 'disabled' | string
}) {
  const colors: Record<string, string> = {
    connected: 'bg-ok-fg',
    error: 'bg-danger-fg',
    connecting: 'bg-warn-fg',
    disabled: 'bg-muted',
  }
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${colors[status] || 'bg-muted'}`}
    />
  )
}

/* ─── Button ─────────────────────────────────────────────────────────── */
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'
const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-white hover:bg-accent-2 active:scale-[0.98]',
  secondary:
    'bg-surface border border-line text-ink hover:bg-surface-2 active:scale-[0.98]',
  ghost: 'text-ink-soft hover:bg-surface-2 hover:text-ink',
  danger:
    'bg-danger-fg text-white hover:opacity-90 active:scale-[0.98]',
}
const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'text-xs px-2.5 py-1.5 gap-1',
  md: 'text-sm px-3.5 py-2 gap-1.5',
  lg: 'text-sm px-5 py-2.5 gap-1.5',
}
export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  children,
  disabled,
  className = '',
  ...rest
}: {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  icon?: LucideIcon
  children: ReactNode
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const IconCmp = loading ? Loader2 : icon
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center font-medium rounded-[var(--radius)] transition-all ${
        BUTTON_VARIANTS[variant]
      } ${BUTTON_SIZES[size]} ${
        disabled || loading ? 'opacity-50 cursor-not-allowed' : ''
      } ${className}`}
    >
      {IconCmp && (
        <IconCmp
          size={size === 'sm' ? 13 : 15}
          strokeWidth={1.5}
          className={loading ? 'animate-spin' : ''}
        />
      )}
      {children}
    </button>
  )
}

/* ─── IconButton（纯图标方按钮，侧边栏/工具栏用）───────────────────────── */
export function IconButton({
  icon: I,
  label,
  size = 'md',
  className = '',
  ...rest
}: {
  icon: LucideIcon
  label: string // aria-label，必填
  size?: 'sm' | 'md'
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      aria-label={label}
      className={`inline-flex items-center justify-center text-ink-soft hover:text-ink hover:bg-surface-2 active:scale-95 rounded-[var(--radius)] transition-all ${
        size === 'sm' ? 'size-7' : 'size-8'
      } ${className}`}
    >
      <I size={size === 'sm' ? 14 : 16} strokeWidth={1.5} />
    </button>
  )
}

/* ─── Input ──────────────────────────────────────────────────────────── */
export function Input({
  label,
  hint,
  error,
  leftIcon,
  className = '',
  ...rest
}: {
  label?: string
  hint?: string
  error?: string
  leftIcon?: LucideIcon
} & InputHTMLAttributes<HTMLInputElement>) {
  const Left = leftIcon
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm text-ink-soft mb-1.5">{label}</label>
      )}
      <div className="relative">
        {Left && (
          <Left
            size={15}
            strokeWidth={1.5}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
          />
        )}
        <input
          {...rest}
          className={`w-full bg-surface border border-line rounded-[var(--radius)] ${
            Left ? 'pl-9' : 'pl-3'
          } pr-3 py-2 text-sm text-ink placeholder:text-muted focus:outline-none focus:border-ink transition-colors ${
            error ? 'border-danger-fg' : ''
          } ${className}`}
        />
      </div>
      {error ? (
        <p className="mt-1 text-xs text-danger-fg">{error}</p>
      ) : hint ? (
        <p className="mt-1 text-xs text-muted">{hint}</p>
      ) : null}
    </div>
  )
}

/* ─── Textarea ───────────────────────────────────────────────────────── */
export function Textarea({
  label,
  hint,
  error,
  className = '',
  ...rest
}: {
  label?: string
  hint?: string
  error?: string
} & TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm text-ink-soft mb-1.5">{label}</label>
      )}
      <textarea
        {...rest}
        className={`w-full bg-surface border border-line rounded-[var(--radius)] px-3 py-2 text-sm text-ink placeholder:text-muted focus:outline-none focus:border-ink transition-colors resize-y min-h-[80px] leading-relaxed ${
          error ? 'border-danger-fg' : ''
        } ${className}`}
      />
      {error ? (
        <p className="mt-1 text-xs text-danger-fg">{error}</p>
      ) : hint ? (
        <p className="mt-1 text-xs text-muted">{hint}</p>
      ) : null}
    </div>
  )
}

/* ─── Select（原生 + ChevronDown）─────────────────────────────────────── */
export function Select({
  label,
  hint,
  className = '',
  children,
  ...rest
}: {
  label?: string
  hint?: string
} & SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm text-ink-soft mb-1.5">{label}</label>
      )}
      <div className="relative">
        <select
          {...rest}
          className={`w-full appearance-none bg-surface border border-line rounded-[var(--radius)] px-3 pr-9 py-2 text-sm text-ink focus:outline-none focus:border-ink transition-colors ${className}`}
        >
          {children}
        </select>
        <ChevronDown
          size={15}
          strokeWidth={1.5}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
        />
      </div>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  )
}

/* ─── Tabs（underline 变体）───────────────────────────────────────────── */
export function Tabs({
  items,
  value,
  onChange,
}: {
  items: { key: string; label: ReactNode; badge?: ReactNode }[]
  value: string
  onChange: (key: string) => void
}) {
  return (
    <div className="flex gap-5 border-b border-line">
      {items.map((it) => {
        const active = it.key === value
        return (
          <button
            key={it.key}
            onClick={() => onChange(it.key)}
            className={`relative pb-2.5 -mb-px border-b-2 transition-colors text-sm ${
              active
                ? 'border-ink text-ink font-medium'
                : 'border-transparent text-muted hover:text-ink-soft'
            }`}
          >
            <span className="inline-flex items-center gap-1.5">
              {it.label}
              {it.badge}
            </span>
          </button>
        )
      })}
    </div>
  )
}

/* ─── SectionHeader（管理页/仪表盘分组标题）────────────────────────────── */
export function SectionHeader({
  title,
  subtitle,
  eyebrow,
  icon: I,
  action,
}: {
  title: string
  subtitle?: string
  eyebrow?: string
  icon?: LucideIcon
  action?: ReactNode
}) {
  return (
    <div className="flex items-end justify-between gap-4 pb-3 border-b border-line">
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-[11px] uppercase tracking-[0.12em] text-muted mb-1">
            {eyebrow}
          </p>
        )}
        <h2 className="font-serif text-xl text-ink tracking-tight flex items-center gap-2">
          {I && <I size={18} strokeWidth={1.5} className="text-ink-soft" />}
          {title}
        </h2>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

/* ─── Skeleton ───────────────────────────────────────────────────────── */
export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`bg-surface-2 animate-pulse rounded-[var(--radius-sm)] ${className}`}
    />
  )
}

/* ─── Tooltip（纯 CSS group-hover）────────────────────────────────────── */
export function Tooltip({
  content,
  children,
}: {
  content: ReactNode
  children: ReactNode
}) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 whitespace-nowrap rounded-[var(--radius-sm)] bg-accent px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100">
        {content}
      </span>
    </span>
  )
}

/* ─── Avatar ─────────────────────────────────────────────────────────── */
export function Avatar({
  name,
  src,
  size = 32,
}: {
  name: string
  src?: string
  size?: number
}) {
  const initial = (name || '?').trim().charAt(0).toUpperCase()
  if (src) {
    return (
      <img
        src={src}
        alt={name}
        style={{ width: size, height: size }}
        className="rounded-full object-cover border border-line"
      />
    )
  }
  return (
    <span
      style={{ width: size, height: size }}
      className="inline-flex items-center justify-center rounded-full bg-surface-2 border border-line text-ink-soft font-serif select-none"
    >
      {initial}
    </span>
  )
}

/* ─── Divider ────────────────────────────────────────────────────────── */
export function Divider({ label }: { label?: string }) {
  if (!label) return <hr className="border-line" />
  return (
    <div className="flex items-center gap-3 text-xs text-muted">
      <hr className="flex-1 border-line" />
      <span>{label}</span>
      <hr className="flex-1 border-line" />
    </div>
  )
}

/* ─── Kbd（键盘键）────────────────────────────────────────────────────── */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex items-center font-mono text-xs bg-canvas border border-line rounded-[var(--radius-sm)] px-1.5 py-0.5 text-ink-soft">
      {children}
    </kbd>
  )
}

/* ─── Pill（中性标签 chip，课程码/文件名用）────────────────────────────── */
export function Pill({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 bg-canvas border border-line rounded-full px-2 py-0.5 text-xs text-ink-soft">
      {children}
    </span>
  )
}

/* ─── CourseIcon（数据 emoji 弱化包装）────────────────────────────────── */
// c.icon / kb.icon 是后端 DB 字段（emoji），前端多处渲染，不能删。
// 这里统一尺寸 + select-none，弱化存在感；未来后端加 icon_name 字段后可换 lucide。
export function CourseIcon({ emoji }: { emoji?: string }) {
  if (!emoji) return null
  return (
    <span className="text-base leading-none select-none" aria-hidden>
      {emoji}
    </span>
  )
}

/* ─── ErrorState ─────────────────────────────────────────────────────── */
export function ErrorState({
  title = '出错了',
  detail,
  onRetry,
}: {
  title?: string
  detail?: string
  onRetry?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center gap-3">
      <AlertTriangle size={32} strokeWidth={1.5} className="text-danger-fg" />
      <div>
        <p className="font-serif text-sm text-ink">{title}</p>
        {detail && <p className="text-xs text-muted mt-1">{detail}</p>}
      </div>
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry}>
          重试
        </Button>
      )}
    </div>
  )
}

/* ─── ToastViewport（toast 单例逻辑在 lib/toast.ts，供 App.tsx 挂载一次）── */
const TOAST_ICONS: Record<ToastVariant, LucideIcon> = {
  success: Check,
  error: AlertTriangle,
  info: Info,
}
const TOAST_ICON_COLOR: Record<ToastVariant, string> = {
  success: 'text-ok-fg',
  error: 'text-danger-fg',
  info: 'text-info-fg',
}

export function ToastViewport() {
  const items = useToasts()
  if (items.length === 0) return null
  return createPortal(
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 w-[min(92vw,360px)]">
      {items.map((t) => {
        const I = TOAST_ICONS[t.variant]
        return (
          <div
            key={t.id}
            className="bg-surface border border-line rounded-[var(--radius)] shadow-[var(--shadow-hover)] px-3.5 py-3 flex items-start gap-2.5 text-sm animate-[toast-in_0.2s_ease-out]"
          >
            <I size={16} strokeWidth={1.5} className={`${TOAST_ICON_COLOR[t.variant]} mt-0.5 shrink-0`} />
            <p className="text-ink leading-snug">{t.message}</p>
            <button
              onClick={() => toast.dismiss(t.id)}
              className="text-muted hover:text-ink ml-auto -mt-0.5 shrink-0"
              aria-label="关闭"
            >
              <X size={14} strokeWidth={1.5} />
            </button>
          </div>
        )
      })}
    </div>,
    document.body,
  )
}
