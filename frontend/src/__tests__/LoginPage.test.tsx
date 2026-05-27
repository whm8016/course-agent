/**
 * LoginPage 组件渲染与交互测试
 * placeholder: "请输入用户名" / "请输入密码" / "至少 4 个字符"（注册模式密码）
 * 按钮文字：登录模式 "登 录"，注册模式 "注 册"
 * 切换按钮："没有账号？点击注册" / "已有账号？点击登录"
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import LoginPage from '../components/pages/LoginPage'

vi.mock('../services/auth', () => ({
  login: vi.fn(),
  register: vi.fn(),
}))

import { login, register } from '../services/auth'

const mockLogin = vi.mocked(login)
const mockRegister = vi.mocked(register)

const MOCK_USER = {
  id: 'u1',
  username: 'alice',
  display_name: 'Alice',
  role: 'student' as const,
  is_admin: false,
}

beforeEach(() => {
  vi.resetAllMocks()
})

// ---------------------------------------------------------------------------
// 渲染测试
// ---------------------------------------------------------------------------

describe('LoginPage 渲染', () => {
  it('默认显示登录表单（用户名 + 密码输入框）', () => {
    render(<LoginPage onLogin={vi.fn()} />)
    expect(screen.getByText('登录以开始学习')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('请输入用户名')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('请输入密码')).toBeInTheDocument()
  })

  it('标题显示"课程学习 Agent"', () => {
    render(<LoginPage onLogin={vi.fn()} />)
    expect(screen.getByText('课程学习 Agent')).toBeInTheDocument()
  })

  it('显示切换到注册的按钮', () => {
    render(<LoginPage onLogin={vi.fn()} />)
    expect(screen.getByText('没有账号？点击注册')).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// 表单验证
// ---------------------------------------------------------------------------

describe('LoginPage 表单验证', () => {
  it('用户名为空时点击提交显示错误提示', async () => {
    render(<LoginPage onLogin={vi.fn()} />)
    // 按钮文字为 "登 录"（含全角空格）
    const submitBtn = screen.getByRole('button', { name: '登 录' })
    await userEvent.click(submitBtn)
    expect(screen.getByText('请输入用户名和密码')).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// 登录流程
// ---------------------------------------------------------------------------

describe('LoginPage 登录流程', () => {
  it('登录成功后调用 onLogin 回调', async () => {
    mockLogin.mockResolvedValue({ token: 'tok', user: MOCK_USER })
    const onLogin = vi.fn()
    render(<LoginPage onLogin={onLogin} />)

    await userEvent.type(screen.getByPlaceholderText('请输入用户名'), 'alice')
    await userEvent.type(screen.getByPlaceholderText('请输入密码'), 'pass1234')
    fireEvent.click(screen.getByRole('button', { name: '登 录' }))

    await waitFor(() => {
      expect(onLogin).toHaveBeenCalledWith(MOCK_USER)
    })
  })

  it('登录失败时显示错误信息', async () => {
    mockLogin.mockRejectedValue(new Error('密码错误'))
    render(<LoginPage onLogin={vi.fn()} />)

    await userEvent.type(screen.getByPlaceholderText('请输入用户名'), 'alice')
    await userEvent.type(screen.getByPlaceholderText('请输入密码'), 'wrongpass')
    fireEvent.click(screen.getByRole('button', { name: '登 录' }))

    await waitFor(() => {
      expect(screen.getByText('密码错误')).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// 注册模式
// ---------------------------------------------------------------------------

describe('LoginPage 注册模式', () => {
  it('点击"没有账号？点击注册"切换到注册表单', async () => {
    render(<LoginPage onLogin={vi.fn()} />)
    await userEvent.click(screen.getByText('没有账号？点击注册'))
    expect(screen.getByText('创建一个新账号')).toBeInTheDocument()
    // 密码框 placeholder 变为 "至少 4 个字符"
    expect(screen.getByPlaceholderText('至少 4 个字符')).toBeInTheDocument()
  })

  it('注册成功后调用 onLogin 回调', async () => {
    mockRegister.mockResolvedValue({ token: 'tok', user: MOCK_USER })
    const onLogin = vi.fn()
    render(<LoginPage onLogin={onLogin} />)

    // 切换到注册模式
    await userEvent.click(screen.getByText('没有账号？点击注册'))

    await userEvent.type(screen.getByPlaceholderText('请输入用户名'), 'alice')
    await userEvent.type(screen.getByPlaceholderText('至少 4 个字符'), 'pass1234')
    fireEvent.click(screen.getByRole('button', { name: '注 册' }))

    await waitFor(() => {
      expect(onLogin).toHaveBeenCalledWith(MOCK_USER)
    })
  })
})
