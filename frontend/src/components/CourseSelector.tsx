import type { Course, KBStatus } from '../types'

interface Props {
  courses: Course[]
  activeCourseId: string
  onSelect: (id: string) => void
}

const STATUS_BADGE: Record<KBStatus, { label: string; cls: string; title: string }> = {
  ready: {
    label: 'RAG',
    cls: 'bg-green-100 text-green-700',
    title: '知识库就绪，可使用 LightRAG 检索作答',
  },
  indexing: {
    label: '索引中',
    cls: 'bg-blue-100 text-blue-700',
    title: '知识库正在索引，完成后会自动启用 RAG',
  },
  pending: {
    label: '未索引',
    cls: 'bg-yellow-100 text-yellow-700',
    title: '知识库已创建，等待管理员上传文件并触发索引',
  },
  paused: {
    label: '已暂停',
    cls: 'bg-orange-100 text-orange-700',
    title: '索引已暂停，可在管理后台继续',
  },
  error: {
    label: '索引出错',
    cls: 'bg-red-100 text-red-700',
    title: '上次索引失败，请到管理后台查看错误并重试',
  },
}

export default function CourseSelector({ courses, activeCourseId, onSelect }: Props) {
  return (
    <div className="space-y-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 px-3 mb-2">
        课程
      </h2>
      {courses.map((c) => {
        const badge = c.kb_status ? STATUS_BADGE[c.kb_status] : null
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm flex items-center gap-2 ${
              activeCourseId === c.id
                ? 'bg-indigo-50 text-indigo-700 font-medium'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <span className="text-lg">{c.icon}</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="truncate">{c.name}</span>
                {badge && (
                  <span
                    className={`text-[10px] leading-none px-1.5 py-0.5 rounded-full whitespace-nowrap ${badge.cls}`}
                    title={badge.title}
                  >
                    {badge.label}
                  </span>
                )}
              </div>
              <div className="text-xs text-slate-400 font-normal truncate">{c.description}</div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
