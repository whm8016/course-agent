/**
 * auth service 单元测试
 * 覆盖：token 存取、isLoggedIn、authHeaders、register/login（fetch mock）
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  getToken,
  getUser,
  saveAuth,
  clearAuth,
  isLoggedIn,
  authHeaders,
} from '../services/auth'
import type { AuthResponse } from '../types'

const MOCK_AUTH: AuthResponse = {
  token: 'test-token-abc',
  user: {
    id: 'user-1',
    username: 'testuser',
    display_name: '测试用户',
    role: 'student',
    is_admin: false,
  },
}

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('getToken', () => {
  it('未登录时返回 null', () => {
    expect(getToken()).toBeNull()
  })

  it('saveAuth 后可读取 token', () => {
    saveAuth(MOCK_AUTH)
    expect(getToken()).toBe('test-token-abc')
  })
})

describe('getUser', () => {
  it('未登录时返回 null', () => {
    expect(getUser()).toBeNull()
  })

  it('saveAuth 后返回正确用户对象', () => {
    saveAuth(MOCK_AUTH)
    const u = getUser()
    expect(u?.username).toBe('testuser')
    expect(u?.id).toBe('user-1')
  })

  it('localStorage 中损坏 JSON 时返回 null', () => {
    localStorage.setItem('auth_user', '{broken json')
    expect(getUser()).toBeNull()
  })
})

describe('isLoggedIn', () => {
  it('未登录返回 false', () => {
    expect(isLoggedIn()).toBe(false)
  })

  it('登录后返回 true', () => {
    saveAuth(MOCK_AUTH)
    expect(isLoggedIn()).toBe(true)
  })

  it('clearAuth 后返回 false', () => {
    saveAuth(MOCK_AUTH)
    clearAuth()
    expect(isLoggedIn()).toBe(false)
  })
})

describe('authHeaders', () => {
  it('未登录时返回空对象', () => {
    expect(authHeaders()).toEqual({})
  })

  it('登录后返回 Bearer 头', () => {
    saveAuth(MOCK_AUTH)
    expect(authHeaders()).toEqual({ Authorization: 'Bearer test-token-abc' })
  })
})

describe('register（fetch mock）', () => {
  it('注册成功时保存 auth 并返回 AuthResponse', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => MOCK_AUTH,
    } as Response)

    const { register } = await import('../services/auth')
    const result = await register('testuser', 'password123')
    expect(result.token).toBe('test-token-abc')
    expect(getToken()).toBe('test-token-abc')
  })

  it('注册失败时抛出错误', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ detail: '用户名已存在' }),
    } as Response)

    const { register } = await import('../services/auth')
    await expect(register('testuser', 'password123')).rejects.toThrow('用户名已存在')
  })
})

describe('login（fetch mock）', () => {
  it('登录成功时保存 auth 并返回 AuthResponse', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => MOCK_AUTH,
    } as Response)

    const { login } = await import('../services/auth')
    const result = await login('testuser', 'password123')
    expect(result.user.username).toBe('testuser')
    expect(getToken()).toBe('test-token-abc')
  })

  it('密码错误时抛出错误', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ detail: '密码错误' }),
    } as Response)

    const { login } = await import('../services/auth')
    await expect(login('testuser', 'wrongpass')).rejects.toThrow('密码错误')
  })
})
