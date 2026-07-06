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
