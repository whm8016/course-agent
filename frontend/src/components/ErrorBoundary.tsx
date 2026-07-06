import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}
interface State {
  hasError: boolean
  error: Error | null
}

/**
 * 全局 React Error Boundary。
 *
 * 组件渲染期抛错时兜底，显示「页面出错了，刷新通常能解决」的 fallback UI，
 * 防止整页白屏、用户无法恢复（此前前端无任何 Error Boundary，渲染异常=直接白屏）。
 *
 * ⚠️ Error Boundary 只捕获【渲染期 / 生命周期】错误，不捕获：
 *  - 事件回调里的错误（onClick 等）
 *  - 异步代码（setTimeout / fetch / SSE / Promise）
 *  这些仍由各自调用方的 try/catch 兜底（如 ChatWindow 的 `出错了: ...`）。
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // 仅控制台留痕，不外发（交付后可接 Sentry 等）
    console.error('[ErrorBoundary] 组件渲染异常:', error, info)
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-canvas flex items-center justify-center p-6">
          <div className="max-w-md w-full bg-surface border border-line rounded-[var(--radius-lg)] p-8 text-center">
            <h1 className="font-serif text-xl text-ink mb-2">页面出错了</h1>
            <p className="text-sm text-ink-soft mb-1">遇到了意外错误，刷新通常能解决。</p>
            {this.state.error && (
              <p className="text-xs text-muted mb-5 font-mono break-all whitespace-pre-wrap">
                {this.state.error.message}
              </p>
            )}
            <button
              onClick={this.handleReload}
              className="px-4 py-2 rounded-[var(--radius)] bg-accent text-white text-sm hover:bg-accent-2 transition"
            >
              刷新页面
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
