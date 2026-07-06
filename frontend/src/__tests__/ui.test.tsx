/**
 * ui.tsx 共享组件单测（Phase B 验收）
 * 覆盖三个代表性组件：
 *   - Button：variant 渲染 / 点击回调 / loading 禁用
 *   - Toggle：checked 态样式 / 受控切换 / disabled 屏蔽
 *   - Badge：默认色 / 旧色名别名映射（5 个现有 import 页面向后兼容的关键）
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Button, Toggle, Badge } from '../components/ui'

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

describe('Button', () => {
  it('渲染 children 文本', () => {
    render(<Button>提交</Button>)
    expect(screen.getByRole('button', { name: '提交' })).toBeInTheDocument()
  })

  it('点击触发 onClick', () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick}>点我</Button>)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('primary 变体应用 accent 黑底类', () => {
    render(<Button>主按钮</Button>)
    expect(screen.getByRole('button').className).toMatch(/bg-accent/)
  })

  it('loading 时禁用且不触发 onClick', () => {
    const onClick = vi.fn()
    render(
      <Button loading onClick={onClick}>
        加载中
      </Button>,
    )
    const btn = screen.getByRole('button')
    expect(btn).toBeDisabled()
    fireEvent.click(btn)
    expect(onClick).not.toHaveBeenCalled()
  })

  it('disabled 属性生效', () => {
    render(<Button disabled>禁用</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })
})

// ---------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------

describe('Toggle', () => {
  it('checked=true 时应用 bg-ink（暖白极简的激活态：黑底，替代旧 indigo）', () => {
    render(<Toggle checked={true} onChange={vi.fn()} />)
    expect(screen.getByRole('button').className).toMatch(/bg-ink/)
  })

  it('checked=false 时应用 bg-line（未激活）', () => {
    render(<Toggle checked={false} onChange={vi.fn()} />)
    expect(screen.getByRole('button').className).toMatch(/bg-line/)
  })

  it('点击调用 onChange 并翻转状态（受控组件）', () => {
    const onChange = vi.fn()
    render(<Toggle checked={false} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('disabled 时不触发 onChange', () => {
    const onChange = vi.fn()
    render(<Toggle checked={false} onChange={onChange} disabled />)
    const btn = screen.getByRole('button')
    fireEvent.click(btn)
    expect(onChange).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// Badge（向后兼容关键：旧色名 slate/green/indigo 必须别名映射到新 5 色）
// ---------------------------------------------------------------------------

describe('Badge 旧色名别名映射', () => {
  it('默认（不传 color）渲染 neutral 色', () => {
    render(<Badge>默认</Badge>)
    expect(screen.getByText('默认').className).toMatch(/bg-neutral-bg/)
  })

  it('旧名 slate → neutral（5 个现有页面向后兼容）', () => {
    render(<Badge color="slate">标签</Badge>)
    expect(screen.getByText('标签').className).toMatch(/bg-neutral-bg/)
  })

  it('旧名 green → ok', () => {
    render(<Badge color="green">就绪</Badge>)
    expect(screen.getByText('就绪').className).toMatch(/bg-ok-bg/)
  })

  it('旧名 indigo → info', () => {
    render(<Badge color="indigo">提示</Badge>)
    expect(screen.getByText('提示').className).toMatch(/bg-info-bg/)
  })

  it('旧名 red → danger', () => {
    render(<Badge color="red">错误</Badge>)
    expect(screen.getByText('错误').className).toMatch(/bg-danger-bg/)
  })

  it('新名 warn 直接生效', () => {
    render(<Badge color="warn">警告</Badge>)
    expect(screen.getByText('警告').className).toMatch(/bg-warn-bg/)
  })

  it('未知色名回退 neutral（防御）', () => {
    render(<Badge color="nonexistent">兜底</Badge>)
    expect(screen.getByText('兜底').className).toMatch(/bg-neutral-bg/)
  })
})
