/**
 * 课程主题图「重建」：确认框必须挂到 body（避免被 overflow 父级裁掉）；
 * POST 入队后由 GET 的 build 字段驱动 5s 轮询——进度条逐块可见、完成 toast、
 * 失败横幅并停止轮询（不再是一句静态"已提交"让用户盲等 20 分钟）。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { toast } from '../lib/toast'
import KbDetailPanel, { type KB } from '../components/pages/KbDetailPanel'

vi.mock('../lib/toast', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), dismiss: vi.fn() },
}))

const GRAPH = {
  course_id: 'c1',
  topic_count: 2,
  edge_count: 1,
  failed_sections: ['实验5（第1块）'],
  topics: [
    {
      topic_id: 't1', label: '元件识别', definition: '识别电阻电容',
      source_section: '大纲', order_idx: 0, safety_weight: 1,
    },
    {
      topic_id: 't2', label: '误差分析', definition: '误差来源',
      source_section: '大纲', order_idx: 1, safety_weight: 1,
    },
  ],
  edges: [
    { src: 't1', dst: 't2', relation: 'prerequisite', confidence: 0.9, evidence: null },
  ],
}

function stubKb(): KB {
  return {
    id: 'kb1',
    course_id: 'c1',
    name: '电路实验',
    description: '',
    icon: '📘',
    system_prompt: '',
    sort_order: 0,
    is_visible: true,
    status: 'ready',
    file_count: 1,
    error_msg: '',
    progress: 100,
    progress_msg: '',
    chunks_done: 10,
    chunks_total: 10,
    token_estimate: 0,
    created_at: 0,
    updated_at: 0,
    builds: [{
      backend: 'lightrag',
      status: 'ready',
      progress: 100,
      progress_msg: '',
      chunks_done: 10,
      chunks_total: 10,
      token_estimate: 0,
      error_msg: '',
      updated_at: 0,
    }],
    files: [],
  }
}

function renderPanel() {
  return render(
    <KbDetailPanel
      kb={stubKb()}
      apiBase="/teacher/courses"
      onDelete={vi.fn()}
      onDeleteFile={vi.fn()}
      onIndex={vi.fn()}
      onPause={vi.fn()}
      onStop={vi.fn()}
      onRefresh={vi.fn()}
      onUploaded={vi.fn()}
      onUpdated={vi.fn()}
    />,
  )
}

// GET 响应队列：>1 条时逐条 shift（mount → POST 后立即查 → 每拍轮询），剩最后一条
// 持续返回（稳态）。POST 恒成功。
let gets: Array<Record<string, unknown>> = []

const isGet = (c: unknown[]) => (c[1] as RequestInit | undefined)?.method !== 'POST'
const getCalls = () => vi.mocked(global.fetch).mock.calls.filter(isGet).length

beforeEach(() => {
  gets = [{ ...GRAPH }]
  vi.clearAllMocks()
  global.fetch = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') {
      return Promise.resolve({
        ok: true,
        json: async () => ({ message: '主题图构建任务已启动', course_id: 'c1', force: true }),
      })
    }
    const next = gets.length > 1 ? gets.shift()! : gets[0]
    return Promise.resolve({ ok: true, json: async () => next })
  }) as typeof fetch
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

/** fake timers 下冲刷微任务链（fetch promise → setState），代替 findBy* 的 interval 轮询 */
async function flush() {
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
}

describe('课程主题图重建', () => {
  it('点击重建弹出确认框（挂在 document.body，不被滚动容器裁掉）', async () => {
    const user = userEvent.setup()
    renderPanel()
    await screen.findByRole('button', { name: '重建' })

    await user.click(screen.getByRole('button', { name: '重建' }))

    const title = await screen.findByText('重建课程主题图')
    expect(title).toBeInTheDocument()
    expect(document.body.contains(title)).toBe(true)
    // 确认层应是 body 的直接子树（portal），不是夹在 overflow 面板里
    const overlay = title.closest('[data-confirm-overlay]')
    expect(overlay).not.toBeNull()
    expect(overlay?.parentElement).toBe(document.body)
  })

  it('确认后 POST force=true，轮询显示逐块进度，完成后 toast 并停止轮询', async () => {
    vi.useFakeTimers()
    gets = [
      { ...GRAPH },                                                    // 挂载首查：无构建
      { ...GRAPH, build: { status: 'extracting', done: 2, total: 5, section: '第3章' } },
      { ...GRAPH, build: null },                                       // 完成一拍
    ]
    renderPanel()
    await flush()
    // fireEvent 而非 userEvent：userEvent 内部 delay 依赖计时器，fake timers 下挂死
    fireEvent.click(screen.getByRole('button', { name: '重建' }))
    fireEvent.click(screen.getByRole('button', { name: '确认重建' }))
    await flush()

    const posts = vi.mocked(global.fetch).mock.calls.filter((c) => !isGet(c))
    expect(String(posts[0][0])).toContain('/api/teacher/courses/c1/topic-graph?force=true')
    // POST 后立即查一次拿到 build：进度条文本可见
    expect(screen.getByText(/正在抽取：第3章/)).toBeInTheDocument()
    expect(screen.getByText('2/5 块')).toBeInTheDocument()
    // 构建中旧图仍在、上一轮 failed_sections 不回显
    expect(screen.getAllByText('元件识别').length).toBeGreaterThan(0)
    expect(screen.queryByText(/抽取失败/)).not.toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })  // 下一拍：完成
    expect(toast.success).toHaveBeenCalledWith('主题图构建完成：2 主题 / 1 边')
    expect(screen.queryByText(/正在抽取/)).not.toBeInTheDocument()

    const after = getCalls()
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })  // 轮询已停
    expect(getCalls()).toBe(after)
  })

  it('GET 报告 error 时显示失败横幅并停止轮询（不 toast 完成）', async () => {
    vi.useFakeTimers()
    gets = [
      { ...GRAPH },
      { ...GRAPH, build: { status: 'error', error: '没有摄入切块审计 JSON' } },
    ]
    renderPanel()
    await flush()
    fireEvent.click(screen.getByRole('button', { name: '重建' }))
    fireEvent.click(screen.getByRole('button', { name: '确认重建' }))
    await flush()

    expect(screen.getByText(/构建失败：没有摄入切块审计 JSON/)).toBeInTheDocument()
    expect(toast.success).not.toHaveBeenCalled()

    const after = getCalls()
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(getCalls()).toBe(after)
  })

  it('刷新页面/重新打开后 GET 带 build 也能接上轮询（queued 态可见）', async () => {
    vi.useFakeTimers()
    gets = [{ ...GRAPH, build: { status: 'queued' } }]
    renderPanel()
    await flush()

    expect(screen.getByText(/已入队，等待后台开始构建/)).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    expect(getCalls()).toBeGreaterThan(1)  // 轮询仍在跑
  })
})
