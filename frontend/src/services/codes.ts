/**
 * 码归一化 / 格式化的前端镜像（与 backend/core/codes.py 保持一致）。
 *
 * - normalizeCode：把用户输入或库内码归一为查库形式（去空白/连字符 + 转大写），
 *   lossless，兼容历史 hex 码。前端拼分享链接、复制课程码时用裸码（归一后）。
 * - formatCode：仅展示用（XXXX-XXXX），DB 与 URL 永不存连字符。
 *
 * 两端各持一份纯函数，避免前端为格式化发请求；改动时需与后端同步。
 */
const DEFAULT_LENGTH = 8

export function normalizeCode(raw: string | null | undefined): string {
  if (!raw) return ''
  return raw.replace(/[\s-]/g, '').toUpperCase()
}

export function formatCode(raw: string | null | undefined): string {
  const n = normalizeCode(raw)
  return n.length === DEFAULT_LENGTH ? `${n.slice(0, 4)}-${n.slice(4)}` : n
}
