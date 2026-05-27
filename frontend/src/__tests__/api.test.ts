/**
 * services/api.ts 接口逻辑单元测试
 * 覆盖：fetchCourses、fetchSessions、createSession、deleteSession
 * 依赖注入：vi.fn() mock global.fetch，不发真实网络请求
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { fetchCourses, fetchSessions, createSession, deleteSession } from '../services/api'

beforeEach(() => {
  localStorage.setItem('auth_token', 'mock-jwt-token')
  vi.resetAllMocks()
})

afterEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// fetchCourses
// ---------------------------------------------------------------------------

describe('fetchCourses', () => {
  it('成功返回课程列表', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ courses: [{ id: 'c1', name: '电路分析', icon: '⚡' }] }),
    } as Response)

    const courses = await fetchCourses()
    expect(courses).toHaveLength(1)
    expect(courses[0].id).toBe('c1')
    // 验证携带了 Authorization 头
    expect(vi.mocked(global.fetch)).toHaveBeenCalledWith(
      '/api/courses',
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: 'Bearer mock-jwt-token' }),
      }),
    )
  })

  it('后端无法连接时抛出中文错误', async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))

    await expect(fetchCourses()).rejects.toThrow('无法连接后端服务')
  })

  it('后端返回非 ok 时抛出 detail 错误', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      text: async () => JSON.stringify({ detail: '未授权' }),
    } as Response)

    await expect(fetchCourses()).rejects.toThrow('未授权')
  })

  it('响应 courses 不是数组时返回空数组', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ courses: null }),
    } as Response)

    const courses = await fetchCourses()
    expect(courses).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// fetchSessions
// ---------------------------------------------------------------------------

describe('fetchSessions', () => {
  it('不带 courseId 时请求 /api/sessions', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ sessions: [] }),
    } as unknown as Response)

    await fetchSessions()
    expect(vi.mocked(global.fetch)).toHaveBeenCalledWith(
      '/api/sessions',
      expect.anything(),
    )
  })

  it('带 courseId 时附加查询参数', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ sessions: [] }),
    } as unknown as Response)

    await fetchSessions('course-xyz')
    expect(vi.mocked(global.fetch)).toHaveBeenCalledWith(
      '/api/sessions?course_id=course-xyz',
      expect.anything(),
    )
  })
})

// ---------------------------------------------------------------------------
// createSession
// ---------------------------------------------------------------------------

describe('createSession', () => {
  it('成功创建会话并返回 Session 对象', async () => {
    const mockSession = {
      id: 's1',
      course_id: 'c1',
      title: '新对话',
      mode: 'chat',
      created_at: 1000,
      updated_at: 1000,
    }
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockSession,
    } as unknown as Response)

    const session = await createSession('c1', '新对话')
    expect(session.id).toBe('s1')
    expect(session.course_id).toBe('c1')
  })

  it('创建失败时抛出错误', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      text: async () => JSON.stringify({ detail: '未选此课程' }),
    } as unknown as Response)

    await expect(createSession('c1')).rejects.toThrow('未选此课程')
  })
})

// ---------------------------------------------------------------------------
// deleteSession
// ---------------------------------------------------------------------------

describe('deleteSession', () => {
  it('发送 DELETE 请求到正确路径', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
    } as unknown as Response)

    await deleteSession('session-abc')
    expect(vi.mocked(global.fetch)).toHaveBeenCalledWith(
      '/api/sessions/session-abc',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })
})
