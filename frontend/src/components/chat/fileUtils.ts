/**
 * 文件类型判定工具。
 * 从 ImageUpload.tsx 拆出，独立 .ts 文件 —— 避免在导出组件的文件里
 * 同时 export 工具函数（触发 react-refresh/only-export-components）。
 */
export function isImage(file: File): boolean {
  return file.type.startsWith('image/')
}
