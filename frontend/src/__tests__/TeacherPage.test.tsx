/**
 * 教师页右下角「用课程码加入课程」悬浮卡收起交互：
 *   - 默认展开；点 ✕ 收起并写入 localStorage
 *   - localStorage 有标记时初始即收起（刷新后不再挡右下角）
 *   - 收起态的小按钮能恢复展开并清掉标记
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import TeacherPage from '../components/pages/TeacherPage'
import type { User } from '../types'

vi.mock('../lib/toast', () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), dismiss: vi.fn() },
}))

const USER: User = { id: 'u1', username: 'teacher1', display_name: '老师', role: 'teacher' }

// 课程列表等接口统一返回空数组：页面骨架渲染即可，不依赖任何数据
function stubFetch() {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [],
  })))
}

beforeEach(() => {
  localStorage.clear()
  stubFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('教师页悬浮入课卡收起', () => {
  it('默认展开，显示课程码输入卡', async () => {
    render(<TeacherPage user={USER} onBack={() => {}} />)
    expect(await screen.findByText('用课程码加入课程')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('XXXXXXXX')).toBeInTheDocument()
  })

  it('点 ✕ 收起：卡片消失、留恢复按钮、localStorage 写入标记', async () => {
    const user = userEvent.setup()
    render(<TeacherPage user={USER} onBack={() => {}} />)
    await user.click(await screen.findByTitle('收起'))

    expect(screen.queryByText('用课程码加入课程')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /加入课程/ })).toBeInTheDocument()
    expect(localStorage.getItem('teacher_join_card_hidden')).toBe('1')
  })

  it('localStorage 有标记时初始即收起（刷新后保持隐藏）', async () => {
    localStorage.setItem('teacher_join_card_hidden', '1')
    render(<TeacherPage user={USER} onBack={() => {}} />)

    expect(screen.getByRole('button', { name: /加入课程/ })).toBeInTheDocument()
    expect(screen.queryByText('用课程码加入课程')).not.toBeInTheDocument()
  })

  it('点恢复按钮：卡片回来、标记清除', async () => {
    localStorage.setItem('teacher_join_card_hidden', '1')
    const user = userEvent.setup()
    render(<TeacherPage user={USER} onBack={() => {}} />)
    await user.click(screen.getByRole('button', { name: /加入课程/ }))

    expect(screen.getByText('用课程码加入课程')).toBeInTheDocument()
    expect(localStorage.getItem('teacher_join_card_hidden')).toBeNull()
  })
})
