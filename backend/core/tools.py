from __future__ import annotations

import json
import logging
from typing import Annotated

from langchain_core.tools import tool

from core.rag import retrieve
from core.llm import _image_to_data_url
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, VISION_MODEL

logger = logging.getLogger(__name__)


@tool
def search_knowledge(
    query: Annotated[str, "用于检索课程知识库的查询文本"],
    course_id: Annotated[str, "课程ID，如 'stamp' 或 'circuit'"],
) -> str:
    """从课程知识库中检索与查询最相关的知识片段。当学生提问需要课程资料支撑时使用此工具。"""
    logger.info("Tool search_knowledge: course=%s query=%s", course_id, query[:60])
    chunks = retrieve(course_id, query, top_k=4)
    if not chunks:
        return json.dumps({"chunks": [], "message": "未找到相关知识片段"}, ensure_ascii=False)
    return json.dumps({"chunks": chunks}, ensure_ascii=False)


@tool
def generate_quiz(
    topic: Annotated[str, "测验主题，如 '基尔霍夫定律'"],
    course_id: Annotated[str, "课程ID"],
    count: Annotated[int, "题目数量，1-5之间"] = 3,
) -> str:
    """基于课程知识点生成测验题目。当学生要求出题、做练习或测验时使用此工具。
    返回的 JSON 包含题目列表，每题有 question, options, answer, explanation 字段。
    注意：此工具仅返回结构化题目指令，实际题目内容由 LLM 在后续步骤中生成。"""
    logger.info("Tool generate_quiz: course=%s topic=%s count=%d", course_id, topic, count)
    chunks = retrieve(course_id, topic, top_k=3)
    context_texts = [c["content"] for c in chunks] if chunks else []
    return json.dumps({
        "action": "generate_quiz",
        "topic": topic,
        "count": min(count, 5),
        "context": context_texts,
    }, ensure_ascii=False)


@tool
def analyze_image(
    image_path: Annotated[str, "图片在服务器上的文件路径"],
    question: Annotated[str, "关于图片的问题或分析要求"] = "请详细分析这张图片。",
) -> str:
    """分析上传的图片内容。用于邮票赏析、电路图识别等场景。"""
    logger.info("Tool analyze_image: path=%s", image_path)
    from openai import OpenAI
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    data_url = _image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        }],
        max_tokens=1024,
    )
    result = response.choices[0].message.content or ""
    return json.dumps({"analysis": result}, ensure_ascii=False)


ALL_TOOLS = [search_knowledge, generate_quiz, analyze_image]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
