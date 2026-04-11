import type { Course } from '../types'

interface Props {
  courses: Course[]
  activeCourseId: string
  onSelect: (id: string) => void
}

export default function CourseSelector({ courses, activeCourseId, onSelect }: Props) {
  return (
    <div className="space-y-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 px-3 mb-2">
        课程
      </h2>
      {courses.map((c) => (
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
          <div>
            <div>{c.name}</div>
            <div className="text-xs text-slate-400 font-normal">{c.description}</div>
          </div>
        </button>
      ))}
    </div>
  )
}
