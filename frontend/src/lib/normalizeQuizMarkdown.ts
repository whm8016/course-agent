/**
 * 将 LLM 输出的 Markdown 预整理，便于 react-markdown + remark-math 正确解析。
 * - 字面上的 \\n 两字符 → 真换行（与真换行可混在一段里）
 * - 全角重音符 ｀ (U+FF40) → 反引号 `（行内代码）
 * - 被换行打碎的 $...$ 行内公式 → 合并为一行
 * - ## 须单独成行、列表与标题的纠正常见问题
 */

const FENCE_SPLIT = /(```[\s\S]*?```)/g

// U+FF40 FULLWIDTH GRAVE ｀（常被误作反引号）
const FULLWIDTH_GRAVE = /\uFF40/g

const PLAIN_MINI_HEADING = new RegExp(
  '^(操作过程|堆数组(?:（层序）)?|堆结构|层序|结论|解析要点|易错点提醒|结构验证)[:：]?\\s*$',
)

/**
 * 非代码块内：把字面上的 \\n、\\r\\n 转成真换行
 */
function unescapeTextLiterals(t: string): string {
  let s = t.replace(/\r\n/g, '\n')
  s = s.replace(/\\r\\n/g, '\n')
  s = s.replace(/\\n/g, '\n')
  s = s.replace(/\\r/g, '\n')
  return s
}

/**
 * 行内 $...$ 里被换行拆碎的公式，合并为单行（避免 remark-math 只识别单行）
 */
function collapseBrokenInlineDollarMathSingle(s: string): string {
  return s.replace(/\$([^$]+?)\$(?!\$)/g, (full, inner: string) => {
    if (!inner || !inner.includes('\n')) return full
    const trim = inner.trim()
    if (/^\d+(\.\d+)?$/.test(trim)) return full
    const collapsed = inner.replace(/\s*\n+\s*/g, ' ').replace(/\s+/g, ' ').trim()
    return `$${collapsed}$`
  })
}

function collapseBrokenInlineDollarMath(segment: string): string {
  if (!segment.includes('$') || !segment.includes('\n')) return segment
  if (segment.includes('$$')) {
    const parts = segment.split('$$')
    return parts
      .map((p, i) => (i % 2 === 1 ? p : collapseBrokenInlineDollarMathSingle(p)))
      .join('$$')
  }
  return collapseBrokenInlineDollarMathSingle(segment)
}

/** 将单独成行的过程小标题（无 #）提升为 ## 标题 */
function promotePlainHeadings(t: string): string {
  return t
    .split('\n')
    .map((line) => {
      const s = line.trim()
      if (!s) return line
      if (s.startsWith('#')) return line
      if (PLAIN_MINI_HEADING.test(s)) {
        return `## ${s.replace(/[:：]\s*$/, '')}`
      }
      return line
    })
    .join('\n')
}

function normalizeSegment(raw: string): string {
  let t = unescapeTextLiterals(raw)
  t = t.replace(FULLWIDTH_GRAVE, '`')
  t = collapseBrokenInlineDollarMath(t)
  t = promotePlainHeadings(t)

  // 句末后紧跟「- 列表项」时，加空行以便解析为列表（否则常与上一段粘成一段）
  t = t.replace(/([。！？；])\n(-\s+\S)/g, '$1\n\n$2')

  t = t.replace(/(参考答案|解析|答案|题解|说明)[:：]\s*(#{1,6}\s)/g, '$1：\n\n$2')
  t = t.replace(/([:：;；)）】])\s*(#{1,6}\s)/g, '$1\n\n$2')
  t = t.replace(/([。！？])\s*(#{1,6}\s)/g, '$1\n\n$2')
  t = t.replace(/^(#{1,6}[^\n]+?)\s+(\d{1,2}\.\s)/gm, '$1\n\n$2')
  t = t.replace(/([。！？])\s*([-*•])\s+/g, '$1\n\n$2 ')
  t = t.replace(/([^\n#])(#{1,6}\s)/g, '$1\n\n$2')

  return t.replace(/\n{3,}/g, '\n\n').trim()
}

export function normalizeQuizMarkdown(content: string): string {
  const s = String(content ?? '')
  if (!s.trim()) return ''
  const parts = s.split(FENCE_SPLIT)
  return parts
    .map((part, i) => (i % 2 === 1 ? part : normalizeSegment(part)))
    .join('')
}
