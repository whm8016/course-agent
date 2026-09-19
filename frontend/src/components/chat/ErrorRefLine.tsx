/**
 * 错误引用号行（错误信封呈现，可观测性重整）。
 *
 * 出错时学生看到的不只是一句文案：错误码 + 8 位编号 + 一键复制。报障从「截图」
 * 变成「报编号」——运维拿编号 GET /admin/errors/recent?ref= 或 grep 后端日志
 * （ContextFilter 已把完整 turn_id/request_id 注入每行）一行命中。
 */
import { useState } from 'react'
import { Copy, Check, AlertTriangle } from 'lucide-react'
import { toast } from '../../lib/toast'

interface Props {
  code: string
  /** 报障引用号。不能叫 `ref`——React 保留 prop 名，lint 按真 ref 访问报错 */
  errorRef: string
}

export default function ErrorRefLine({ code, errorRef }: Props) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    const text = `错误编号: ${errorRef}\n错误码: ${code}\n时间: ${new Date().toISOString()}`
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast.success('错误信息已复制，请发给老师/管理员')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error(`复制失败，请手动抄下编号：${errorRef}`)
    }
  }

  return (
    <div className="mt-2 flex items-center gap-2 flex-wrap text-xs">
      <span className="inline-flex items-center gap-1 text-danger-fg">
        <AlertTriangle size={12} strokeWidth={1.5} />
        <span className="font-mono">{code}</span>
      </span>
      {errorRef && <span className="text-muted">编号 <span className="font-mono">{errorRef}</span></span>}
      <button
        type="button"
        onClick={copy}
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[var(--radius)] border border-line text-muted hover:text-ink hover:border-ink/40 transition-colors"
        aria-label="复制错误信息"
      >
        {copied ? <Check size={12} strokeWidth={1.5} /> : <Copy size={12} strokeWidth={1.5} />}
        复制错误信息
      </button>
    </div>
  )
}
