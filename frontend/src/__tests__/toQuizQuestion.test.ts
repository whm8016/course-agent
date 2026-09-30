/** 判断题枚举对齐回归：后端判断题键是 concept（无 options、answer true/false）。 */
import { describe, expect, it } from 'vitest'

import { toQuizQuestion } from '../components/chat/ChatWindow'

describe('toQuizQuestion 判断题归一', () => {
  it('concept 且无 options → A/B 判断卡，true 归 A', () => {
    const q = toQuizQuestion({
      question: '叠加定理适用于线性电路',
      question_type: 'concept',
      correct_answer: 'true',
      explanation: '线性电路才行',
    })
    expect(q.options).toEqual(['A. 正确', 'B. 错误'])
    expect(q.answer).toBe('A')
  })

  it('concept 的 false / 中文答案 → B', () => {
    const q = toQuizQuestion({
      question_type: 'concept',
      correct_answer: 'false',
    })
    expect(q.answer).toBe('B')
    const q2 = toQuizQuestion({
      question_type: 'concept',
      correct_answer: '错误',
    })
    expect(q2.answer).toBe('B')
  })

  it('历史存量键 true_false 同样归一到判断卡（不因旧数据退化）', () => {
    const q = toQuizQuestion({ question_type: 'true_false', correct_answer: 'true' })
    expect(q.options).toEqual(['A. 正确', 'B. 错误'])
    expect(q.answer).toBe('A')
  })

  it('choice（带 options）不受判断题分支影响', () => {
    const q = toQuizQuestion({
      question_type: 'choice',
      options: { A: '1Ω', B: '2Ω', C: '3Ω', D: '4Ω' },
      correct_answer: 'B',
    })
    expect(q.options).toEqual(['A. 1Ω', 'B. 2Ω', 'C. 3Ω', 'D. 4Ω'])
    expect(q.answer).toBe('B')
  })

  it('short_answer（无 options 的非 concept）仍走简答分支', () => {
    const q = toQuizQuestion({
      question_type: 'short_answer',
      correct_answer: '参考答案',
    })
    expect(q.options).toEqual([])
    expect(q.answer).toBe('参考答案')
  })
})
