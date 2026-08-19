/**
 * VoiceInput 语音输入单测（B3）。
 * 覆盖：能力缺失不渲染 / 点击开始录音（红态+计时）/ 停止后调 transcribeAudio
 * 且文字经 onTranscribed 回填 / 转写 503（未配置）隐藏按钮 / 权限失败不炸。
 * MediaRecorder 与 getUserMedia 全部 mock（jsdom 无真实音频栈）。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, fireEvent, screen, waitFor } from '@testing-library/react'
import VoiceInput from '../components/chat/VoiceInput'
import { transcribeAudio } from '../services/api'

vi.mock('../services/api', () => ({
  transcribeAudio: vi.fn(),
}))
vi.mock('../lib/toast', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}))

/** 可控的假 MediaRecorder：stop 触发 onstop（模拟录音数据就绪） */
class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = []
  static supportedTypes = new Set(['audio/webm;codecs=opus'])
  state = 'inactive'
  stream: MediaStream
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(stream: MediaStream) {
    this.stream = stream
    FakeMediaRecorder.instances.push(this)
  }
  static isTypeSupported(t: string) {
    return FakeMediaRecorder.supportedTypes.has(t)
  }
  start() {
    this.state = 'recording'
  }
  stop() {
    if (this.state === 'inactive') return
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['audio-chunk'], { type: 'audio/webm' }) })
    this.onstop?.()
  }
}

function fakeStream(): MediaStream {
  return {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream
}

const fakeGetUserMedia = vi.fn().mockResolvedValue(fakeStream())

function stubNavigator(mediaDevices: unknown) {
  vi.stubGlobal('navigator', { ...window.navigator, mediaDevices })
}

async function clickMic() {
  fireEvent.click(screen.getByRole('button', { name: '语音输入' }))
  await vi.waitFor(() => {
    expect(FakeMediaRecorder.instances.length).toBe(1)
  })
}

describe('VoiceInput 语音输入', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    FakeMediaRecorder.instances = []
    FakeMediaRecorder.supportedTypes = new Set(['audio/webm;codecs=opus'])
    vi.stubGlobal('MediaRecorder', FakeMediaRecorder as unknown as typeof MediaRecorder)
    fakeGetUserMedia.mockReset()
    fakeGetUserMedia.mockResolvedValue(fakeStream())
    stubNavigator({ getUserMedia: fakeGetUserMedia })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('无 MediaRecorder（老 WebView）→ 不渲染', () => {
    vi.stubGlobal('MediaRecorder', undefined)
    const { container } = render(<VoiceInput onTranscribed={vi.fn()} />)
    expect(container.querySelector('button')).toBeNull()
  })

  it('点击开始录音 → 停止 → 转写文字回填', async () => {
    vi.mocked(transcribeAudio).mockResolvedValue({
      text: '今天讲导数',
      model: 'paraformer-v2',
      duration_ms: 12,
    })
    const onTranscribed = vi.fn()
    render(<VoiceInput onTranscribed={onTranscribed} />)

    await clickMic()

    // 录音中：红态停止按钮 + 计时
    const stopBtn = screen.getByRole('button', { name: '停止录音' })
    fireEvent.click(stopBtn)

    await waitFor(() => {
      expect(transcribeAudio).toHaveBeenCalledTimes(1)
    })
    const sent = vi.mocked(transcribeAudio).mock.calls[0][0]
    expect(sent.name).toBe('speech.webm') // mime→扩展名对齐（webm;codecs=opus → .webm）
    await waitFor(() => {
      expect(onTranscribed).toHaveBeenCalledWith('今天讲导数')
    })
  })

  it('转写 503（未配置 ASR）→ 隐藏按钮', async () => {
    vi.mocked(transcribeAudio).mockRejectedValue(new Error('未配置语音识别模型'))
    const { container } = render(<VoiceInput onTranscribed={vi.fn()} />)

    await clickMic()
    fireEvent.click(screen.getByRole('button', { name: '停止录音' }))

    await waitFor(() => {
      expect(container.querySelector('button')).toBeNull() // 入口隐藏
    })
  })

  it('麦克风权限失败 → toast 提示且不渲染录音态', async () => {
    fakeGetUserMedia.mockRejectedValueOnce(new Error('denied'))
    const { container } = render(<VoiceInput onTranscribed={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: '语音输入' }))
    await waitFor(() => {
      expect(fakeGetUserMedia).toHaveBeenCalled()
    })
    // 回到 idle 态（还是麦克风按钮），没有停止录音按钮
    expect(screen.getByRole('button', { name: '语音输入' })).toBeTruthy()
    expect(container.textContent).not.toContain('停止')
  })

  it('mimeType 降级到 audio/mp4（iOS Safari）→ 文件名 .m4a', async () => {
    FakeMediaRecorder.supportedTypes = new Set(['audio/mp4'])
    vi.mocked(transcribeAudio).mockResolvedValue({ text: 'x', model: 'm', duration_ms: 1 })
    render(<VoiceInput onTranscribed={vi.fn()} />)

    await clickMic()
    fireEvent.click(screen.getByRole('button', { name: '停止录音' }))

    await waitFor(() => {
      expect(vi.mocked(transcribeAudio).mock.calls[0][0].name).toBe('speech.m4a')
    })
  })
})
