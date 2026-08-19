import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Mic, Square } from 'lucide-react'
import { transcribeAudio } from '../../services/api'
import { toast } from '../../lib/toast'

/**
 * 语音输入按钮（B 线）：点击开始录音（红色脉动 + 计时），再点停止 / 60s 自动停，
 * 录完 POST /api/transcribe 转写，文字经 onTranscribed 回填输入框（不自动发送）。
 *
 * 优雅降级三层：
 * - 无 getUserMedia / MediaRecorder（老 WebView）→ 整体不渲染
 * - 后端未配置 ASR（503）→ 首次点击后隐藏按钮
 * - mimeType 能力：webm;codecs=opus（Chrome/Android）→ audio/mp4（iOS Safari）
 */
const MAX_RECORD_SEC = 60

// 显式 MIME→扩展名（与后端 api/transcribe._MIME_EXT 对齐；后端按 content-type 判定）
const MIME_CANDIDATES: { mime: string; ext: string }[] = [
  { mime: 'audio/webm;codecs=opus', ext: 'webm' },
  { mime: 'audio/webm', ext: 'webm' },
  { mime: 'audio/mp4', ext: 'm4a' },
]

function pickMime(): { mime: string; ext: string } | null {
  if (typeof MediaRecorder === 'undefined') return null
  for (const c of MIME_CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(c.mime)) return c
    } catch {
      /* isTypeSupported 个别实现会抛，继续下一个 */
    }
  }
  return null
}

interface Props {
  onTranscribed: (text: string) => void
  disabled?: boolean
}

export default function VoiceInput({ onTranscribed, disabled }: Props) {
  const [phase, setPhase] = useState<'idle' | 'recording' | 'transcribing'>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [hidden, setHidden] = useState(false) // 后端 503 → 隐藏入口
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const mimeRef = useRef<{ mime: string; ext: string } | null>(null)

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') {
      try {
        rec.stop()
      } catch {
        /* 已停止 */
      }
    }
    recorderRef.current = null
  }, [])

  // 卸载时停录音关轨道（防 WebSocket/轨道泄漏）
  useEffect(
    () => () => {
      cleanup()
      recorderRef.current?.stream.getTracks().forEach((t) => t.stop())
    },
    [cleanup],
  )

  const transcribe = useCallback(
    async (blob: Blob) => {
      const { ext } = mimeRef.current ?? { ext: 'webm' }
      setPhase('transcribing')
      try {
        const r = await transcribeAudio(new File([blob], `speech.${ext}`))
        if (r.text.trim()) onTranscribed(r.text.trim())
      } catch (e) {
        const msg = e instanceof Error ? e.message : '语音识别失败'
        // 后端未配置 ASR：隐藏入口（此后不再打扰）；其余错误 toast 一次
        if (/未配置/.test(msg)) {
          setHidden(true)
        } else {
          toast.error(msg)
        }
      } finally {
        setPhase('idle')
        setElapsed(0)
      }
    },
    [onTranscribed],
  )

  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setHidden(true)
      return
    }
    const mime = pickMime()
    if (!mime) {
      toast.error('当前浏览器不支持录音')
      return
    }
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      toast.error('无法访问麦克风（检查浏览器权限，需 https 环境）')
      return
    }
    mimeRef.current = mime
    chunksRef.current = []
    const rec = new MediaRecorder(stream, { mimeType: mime.mime })
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      const blob = new Blob(chunksRef.current, { type: mime.mime })
      if (blob.size > 0) void transcribe(blob)
      else setPhase('idle')
    }
    recorderRef.current = rec
    rec.start()
    setPhase('recording')
    setElapsed(0)
    timerRef.current = setInterval(() => {
      setElapsed((s) => {
        if (s + 1 >= MAX_RECORD_SEC) {
          // 60s 自动停（清理在 onstop 链路里收尾）
          cleanup()
          return MAX_RECORD_SEC
        }
        return s + 1
      })
    }, 1000)
  }, [cleanup, transcribe])

  // 能力检测在渲染期做（不是点击时）：无 getUserMedia / MediaRecorder 的老 WebView
  // 直接不渲染按钮，移动端用户看不到死入口
  const supported =
    typeof MediaRecorder !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
  if (!supported || hidden) return null

  if (phase === 'recording') {
    return (
      <button
        type="button"
        onClick={cleanup}
        title={`停止录音（${MAX_RECORD_SEC}s 上限）`}
        aria-label="停止录音"
        className="inline-flex items-center gap-1 px-2 py-2 rounded-[var(--radius)] text-danger-fg border border-danger-fg transition shrink-0 animate-pulse"
      >
        <Square size={16} strokeWidth={1.5} />
        <span className="text-[11px] tabular-nums">{elapsed}s</span>
      </button>
    )
  }

  return (
    <button
      type="button"
      onClick={() => void startRecording()}
      disabled={disabled || phase === 'transcribing'}
      title={phase === 'transcribing' ? '转写中…' : '语音输入（点击开始录音）'}
      aria-label="语音输入"
      className="p-2 rounded-[var(--radius)] text-muted hover:text-ink hover:bg-surface-2 transition shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {phase === 'transcribing' ? (
        <Loader2 size={20} strokeWidth={1.5} className="animate-spin" />
      ) : (
        <Mic size={20} strokeWidth={1.5} />
      )}
    </button>
  )
}
