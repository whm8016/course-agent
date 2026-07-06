import type { LucideIcon } from 'lucide-react'
import { ArrowLeft } from 'lucide-react'

export interface HubItem {
  icon: LucideIcon
  label: string
  desc: string
  onClick: () => void
  /** 右上角小徽标（如未读通知数）*/
  badge?: number
}

interface Props {
  title: string
  icon: LucideIcon
  subtitle?: string
  items: HubItem[]
  onBack: () => void
}

/**
 * 聚合入口页（工作台 / 设置 共用）。
 * 侧边栏只留 2 个入口 → 点开本页 → 卡片网格列出子功能 → 点卡片再进对应 overlay。
 * 中转层：避免侧边栏摊十几个按钮，又不必为每个功能嵌套菜单。
 */
export default function HubPage({ title, icon: Icon, subtitle, items, onBack }: Props) {
  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="px-6 py-4 bg-surface border-b border-line flex items-center gap-3">
        <button
          onClick={onBack}
          className="text-ink-soft hover:text-ink text-sm transition flex items-center gap-1"
        >
          <ArrowLeft size={16} strokeWidth={1.5} />
          返回
        </button>
        <h1 className="font-serif text-lg text-ink flex items-center gap-2">
          <Icon size={18} strokeWidth={1.5} className="text-ink-soft" />
          {title}
        </h1>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {subtitle && <p className="text-xs text-muted mb-4">{subtitle}</p>}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {items.map((it) => {
              const ItemIcon = it.icon
              return (
                <button
                  key={it.label}
                  onClick={it.onClick}
                  className="group relative text-left p-4 rounded-[var(--radius-lg)] border border-line bg-surface hover:border-ink/30 hover:bg-surface-2 transition"
                >
                  <div className="flex items-center gap-3 mb-1">
                    <span className="w-9 h-9 shrink-0 rounded-[var(--radius)] bg-surface-2 flex items-center justify-center text-ink-soft group-hover:text-ink transition">
                      <ItemIcon size={18} strokeWidth={1.5} />
                    </span>
                    <span className="font-medium text-ink">{it.label}</span>
                    {it.badge !== undefined && it.badge > 0 && (
                      <span className="ml-auto inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] bg-danger-fg text-white rounded-full">
                        {it.badge > 99 ? '99+' : it.badge}
                      </span>
                    )}
                  </div>
                  {it.desc && <p className="text-xs text-muted leading-relaxed pl-12">{it.desc}</p>}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
