/**
 * 知识库（KB）相关的共享常量与格式化工具。
 * 从 KbDetailPanel.tsx 拆出，独立 .ts 文件 —— 避免在导出组件的文件里
 * 同时 export 运行时常量/函数（触发 react-refresh/only-export-components）。
 */

export const STATUS_LABEL: Record<string, string> = {
  pending: '待索引',
  indexing: '索引中...',
  ready: '就绪',
  error: '错误',
  paused: '已暂停',
}

export const STATUS_COLOR: Record<string, string> = {
  pending: 'bg-warn-bg text-warn-fg',
  indexing: 'bg-info-bg text-info-fg',
  ready: 'bg-ok-bg text-ok-fg',
  error: 'bg-danger-bg text-danger-fg',
  paused: 'bg-warn-bg text-warn-fg',
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleString('zh-CN')
}
