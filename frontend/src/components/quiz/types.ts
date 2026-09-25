/**
 * Quiz 配置的类型与默认值。
 * 从 QuizConfigPanel.tsx 拆出，独立 .ts 文件 —— 避免在导出组件的文件里
 * 同时 export 运行时常量（触发 react-refresh/only-export-components）。
 */
export interface QuizConfig {
  topic: string
  count: number
  difficulty: string
  questionType: string
  preference: string
}

export const DEFAULT_QUIZ_CONFIG: QuizConfig = {
  topic: '',
  count: 3,
  difficulty: '',
  questionType: '',
  preference: '',
}

/** 题型枚举值 → 中文标签（单一真相源：QuizConfigPanel 的选项与 ChatWindow 的
 *  请求气泡回显共用；键必须与后端 _VALID_TYPES 词表对齐，判断题=concept）。 */
export const QUESTION_TYPE_LABELS: Record<string, string> = {
  choice: '选择题',
  concept: '判断题',
  short_answer: '简答题',
}
