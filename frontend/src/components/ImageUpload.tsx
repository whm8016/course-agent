import { useRef } from 'react'
import { FiImage, FiX } from 'react-icons/fi'

interface Props {
  preview: string | null
  onSelect: (file: File) => void
  onClear: () => void
}

export default function ImageUpload({ preview, onSelect, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onSelect(file)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="relative inline-flex">
      {preview ? (
        <div className="relative">
          <img src={preview} alt="预览" className="h-16 w-16 object-cover rounded-lg border border-slate-200" />
          <button
            onClick={onClear}
            className="absolute -top-1.5 -right-1.5 bg-red-500 text-white rounded-full p-0.5 hover:bg-red-600 transition"
          >
            <FiX size={12} />
          </button>
        </div>
      ) : (
        <button
          onClick={() => inputRef.current?.click()}
          className="p-2 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition"
          title="上传图片"
        >
          <FiImage size={20} />
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  )
}
