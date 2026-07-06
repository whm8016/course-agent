import type { Course, KBStatus } from '../../types'
import { CourseIcon } from '../ui'

interface Props {
  courses: Course[]
  activeCourseId: string
  onSelect: (id: string) => void
}

const STATUS_BADGE: Record<KBStatus, { label: string; cls: string; title: string }> = {
  ready: {
    label: 'RAG',
    cls: 'bg-ok-bg text-ok-fg',
    title: '知识库就绪，可使用 LightRAG 检索作答',
  },
  indexing: {
    label: '索引中',
    cls: 'bg-info-bg text-info-fg',
    title: '知识库正在索引，完成后会自动启用 RAG',
  },
  pending: {
    label: '未索引',
    cls: 'bg-warn-bg text-warn-fg',
    title: '知识库已创建，等待管理员上传文件并触发索引',
  },
  paused: {
    label: '已暂停',
    cls: 'bg-warn-bg text-warn-fg',
    title: '索引已暂停，可在管理后台继续',
  },
  error: {
    label: '索引出错',
    cls: 'bg-danger-bg text-danger-fg',
    title: '上次索引失败，请到管理后台查看错误并重试',
  },
}

export default function CourseSelector({ courses, activeCourseId, onSelect }: Props) {
  return (
    <div className="space-y-1">
      <h2 className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted px-3 mb-2">
        课程
      </h2>
      {courses.map((c) => {
        const badge = c.kb_status ? STATUS_BADGE[c.kb_status] : null
        const active = activeCourseId === c.id
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`w-full text-left px-3 py-2.5 rounded-[var(--radius)] transition-colors text-sm flex items-center gap-2.5 ${
              active
                ? 'bg-surface-2 text-ink font-medium'
                : 'text-ink-soft hover:bg-surface-2 hover:text-ink'
            }`}
          >
            <CourseIcon emoji={c.icon} />
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
              {c.description && (
                <div className="text-xs text-muted font-normal truncate">{c.description}</div>
              )}
            </div>
          </button>
        )
      })}
    </div>
  )
}
