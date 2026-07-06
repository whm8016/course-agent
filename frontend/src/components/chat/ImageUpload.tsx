import { useRef } from 'react'
import { Image as ImageIcon, X, File as FileIcon } from 'lucide-react'

export interface PendingFile {
  file: File
  preview: string // 图片：ObjectURL；文档：空串
  kind: 'image' | 'doc'
  name: string
}

interface Props {
  files: PendingFile[]
  onSelect: (file: File) => void
  onRemove: (index: number) => void
}

export default function ImageUpload({ files, onSelect, onRemove }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files
    if (list) {
      Array.from(list).forEach((f) => onSelect(f))
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {files.map((pf, i) => (
        <div key={i} className="relative">
          {pf.kind === 'image' && pf.preview ? (
            <img
              src={pf.preview}
              alt={pf.name}
              className="h-12 w-12 object-cover rounded-[var(--radius)] border border-line"
            />
          ) : (
            <div className="h-12 px-2 flex items-center gap-1 rounded-[var(--radius)] border border-line bg-surface-2 text-[11px] text-ink-soft max-w-[140px]">
              <FileIcon size={14} strokeWidth={1.5} className="shrink-0 text-muted" />
              <span className="truncate">{pf.name}</span>
            </div>
          )}
          <button
            onClick={() => onRemove(i)}
            className="absolute -top-1.5 -right-1.5 bg-danger-fg text-white rounded-full p-0.5 hover:opacity-90 transition"
            aria-label="移除附件"
          >
            <X size={12} strokeWidth={1.5} />
          </button>
        </div>
      ))}
      <button
        onClick={() => inputRef.current?.click()}
        className="p-2 rounded-[var(--radius)] text-muted hover:text-ink hover:bg-surface-2 transition shrink-0"
        title="上传图片或文档（支持 PDF/Word/TXT/代码等）"
        aria-label="上传图片或文档"
      >
        <ImageIcon size={20} strokeWidth={1.5} />
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.pdf,.txt,.md,.docx,.doc,.xlsx,.pptx,.csv,.json,.py,.js,.ts"
        multiple
        className="hidden"
        onChange={handleChange}
      />
    </div>
  )
}
