import { useRef } from 'react'
import { FiImage, FiX, FiFile } from 'react-icons/fi'

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

function isImage(file: File): boolean {
  return file.type.startsWith('image/')
}

export { isImage }

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
            <img src={pf.preview} alt={pf.name} className="h-12 w-12 object-cover rounded-lg border border-slate-200" />
          ) : (
            <div className="h-12 px-2 flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 text-[11px] text-slate-600 max-w-[140px]">
              <FiFile size={14} className="shrink-0 text-slate-400" />
              <span className="truncate">{pf.name}</span>
            </div>
          )}
          <button
            onClick={() => onRemove(i)}
            className="absolute -top-1.5 -right-1.5 bg-red-500 text-white rounded-full p-0.5 hover:bg-red-600 transition"
          >
            <FiX size={12} />
          </button>
        </div>
      ))}
      <button
        onClick={() => inputRef.current?.click()}
        className="p-2 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition shrink-0"
        title="上传图片或文档（支持 PDF/Word/TXT/代码等）"
      >
        <FiImage size={20} />
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
