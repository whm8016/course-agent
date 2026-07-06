import { useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { formatCode, normalizeCode } from '../../services/codes'
import type { KB } from './KbDetailPanel'

interface Props {
  kb: KB
  /** 提供「生成/重置课程码」按钮；不传则为只读展示（如管理后台）。 */
  onReset?: () => void
  resetting?: boolean
}

/** 课程码分享区块：课程码 + 二维码 + 分享链接 + 复制（+ 可选重置）。
 *
 * 抽离自 TeacherPage，供教师工作台与管理后台共用，确保两处「学生入课方式」
 * 展示一致。放在详情区顶部，避免码被埋在长详情底部找不到。
 */
export default function JoinCodeShareSection({ kb, onReset, resetting }: Props) {
  const [copied, setCopied] = useState(false)

  const copy = (text: string) => {
    try {
      navigator.clipboard?.writeText(text)
    } catch {
      /* 剪贴板被禁用时静默失败，不打断交互 */
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const joinUrl = `${window.location.origin}/join/${normalizeCode(kb.join_code)}`

  return (
    <section className="bg-surface rounded-[var(--radius-lg)] border border-line p-6">
      <h3 className="font-semibold text-ink mb-4 flex items-center gap-2">
        <span></span> 学生入课方式
      </h3>
      {kb.join_code ? (
        <div className="grid md:grid-cols-[auto_1fr] gap-6 items-start">
          {/* 左：课程码展示 + 二维码（扫码入课） */}
          <div className="flex flex-col items-center gap-3">
            <span className="text-3xl font-mono font-bold tracking-widest text-ink bg-surface-2 px-6 py-3 rounded-[var(--radius)] border border-line">
              {formatCode(kb.join_code)}
            </span>
            <QRCodeSVG
              value={joinUrl}
              size={140}
              level="M"
              className="rounded-[var(--radius)] border border-line p-2 bg-surface"
            />
            <span className="text-xs text-muted">扫码即可入课</span>
          </div>
          {/* 右：分享链接 + 复制/重置操作 */}
          <div className="flex flex-col gap-3 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => copy(normalizeCode(kb.join_code))}
                className="px-4 py-2 text-sm bg-surface-2 hover:bg-surface-2 rounded-[var(--radius)] transition"
              >
                {copied ? '已复制!' : '复制课程码'}
              </button>
              {onReset && (
                <button
                  onClick={onReset}
                  disabled={resetting}
                  className="px-4 py-2 text-sm bg-warn-bg text-warn-fg hover:bg-warn-bg rounded-[var(--radius)] transition disabled:opacity-50"
                >
                  {resetting ? '生成中...' : '重置课程码'}
                </button>
              )}
            </div>
            <p className="text-xs text-muted">
              学生可通过 ① 输入课程码 ② 点击分享链接 ③ 扫描二维码 三种方式加入。
              {onReset && '重置后旧码立即失效。'}
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted">暂无课程码</span>
          {onReset && (
            <button
              onClick={onReset}
              disabled={resetting}
              className="px-4 py-2 text-sm bg-accent text-white rounded-[var(--radius)] hover:bg-accent-2 transition disabled:opacity-50"
            >
              {resetting ? '生成中...' : '生成课程码'}
            </button>
          )}
        </div>
      )}
    </section>
  )
}
