from __future__ import annotations

from core.cache import cache_delete, cache_get, cache_set

ROUTER_PROMPT = """你是一个智能路由器，负责分析学生消息的意图并分类。

你服务的是一个**特定课程**的学习助教系统。学生的问题必须与所学课程内容相关。

根据学生的消息，判断属于以下哪种意图：
- "teach": 学生在提问、请求讲解某个知识点、寻求帮助理解概念（且与课程相关）
- "quiz": 学生要求出题、做练习、测验、考考我、来几道题
- "summarize": 学生要求总结、归纳、回顾已学内容、生成知识点清单
- "vision": 学生上传了图片需要分析（此项由系统自动判断，你不需要处理）
- "off_topic": 学生提出的问题与课程学习完全无关，例如政治、娱乐八卦、生活闲聊、与课程无关的知识等。简单的问候语（你好、谢谢等）不算 off_topic，应归为 teach。

判断 off_topic 时要宽松一些：如果问题可能与课程沾边（比如算法课被问"什么是人工智能"），应归为 teach 而非 off_topic。只有明显完全无关的问题才归为 off_topic。

只输出一个 JSON 对象，格式为：{{"intent": "teach"}}
不要输出其他任何内容。"""

QUIZ_PROMPT = """你是一个课程测验出题专家。根据提供的课程知识内容，生成高质量的测验题。

要求：
1. 题目紧扣知识内容，考察核心概念
2. 选项设计合理，干扰项具有迷惑性
3. 提供详细的答案解析
4. 难度适中，适合初学者

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{
  "questions": [
    {{
      "question": "题目内容",
      "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
      "answer": "A",
      "explanation": "解析内容"
    }}
  ]
}}"""

SUMMARY_PROMPT = """你是一个学习总结专家。根据对话历史，为学生生成一份结构清晰的学习小结。

要求：
1. 提炼对话中涉及的核心知识点
2. 用简洁的条目列出要点
3. 标注学生可能还需要加强的地方
4. 给出下一步学习建议
5. 使用 Markdown 格式，结构清晰"""

_FALLBACK_PROMPT = "你是一个通用学习助手。请尽力回答学生与课程学习相关的问题。如果问题与课程学习完全无关，请礼貌拒绝。"

_PROMPT_CACHE_KEY = "course:prompt:{}"
_PROMPT_CACHE_TTL = 600  # 10 分钟


async def get_course_prompt(course_id: str) -> str:
    """从 Redis 缓存或数据库获取课程 system_prompt。"""
    key = _PROMPT_CACHE_KEY.format(course_id)
    cached = await cache_get(key)
    if cached is not None:
        return cached

    from sqlalchemy import select
    from core.database import AsyncSessionLocal, KnowledgeBase

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeBase.system_prompt).where(KnowledgeBase.course_id == course_id)
        )
        row = result.first()

    prompt = (row[0] or "").strip() if row else ""
    if not prompt:
        prompt = _FALLBACK_PROMPT

    await cache_set(key, prompt, ttl=_PROMPT_CACHE_TTL)
    return prompt


async def invalidate_course_prompt_cache(course_id: str) -> None:
    """管理员更新 system_prompt 后调用，使缓存立即失效。"""
    await cache_delete(_PROMPT_CACHE_KEY.format(course_id))
