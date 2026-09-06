"""6 个跨域评测任务：每题需要 2-3 个不同领域工具协作才能完成。

prompt 用自然语言描述需求（不点名工具），expected_tools 用 (server, tool) 声明
打分标准，经 wrapped_tool_name 计算 wrapped 名——与合成池的命名规则同源，
改池子命名时这里不会漂移。
"""
from __future__ import annotations

from core.mcp.adapter import wrapped_tool_name

_TASK_DEFS: list[dict] = [
    {
        "id": "paper_translate",
        "prompt": (
            "我在写课程调研报告：帮我检索一下 transformer 注意力机制方向最近的学术论文，"
            "挑一篇代表性的，把它的英文摘要翻译成中文给我。"
        ),
        "expect": [("arxiv", "search"), ("translate", "text")],
    },
    {
        "id": "tokyo_trip",
        "prompt": (
            "下周想去东京玩：帮我查一下东京未来五天的天气预报，再看看现在日元兑人民币的"
            "汇率是多少，好决定换多少现金。"
        ),
        "expect": [("weather", "forecast"), ("forex", "get_rate")],
    },
    {
        "id": "ai_edu_news",
        "prompt": (
            "老师让我了解 AI 教育的舆论环境：检索一下最近人工智能进课堂的新闻，分析一下"
            "这个话题的舆论情感倾向，最后把结果画成一张柱状图。"
        ),
        "expect": [("news", "search"), ("news", "sentiment_analysis"), ("chart", "render_chart")],
    },
    {
        "id": "repo_audit",
        "prompt": (
            "我想评估一个开源项目值不值得引入：查一下 langgen instructor 仓库的基本统计"
            "（star/fork 之类），再搜搜它有没有关于内存泄漏的 issue，最后把调研材料打包"
            "成一个 zip 存到本地。"
        ),
        "expect": [("github", "repo_stats"), ("github", "issue_search"), ("file", "compress")],
    },
    {
        "id": "hike_plan",
        "prompt": (
            "周末想组织一次班级徒步：先查查周六郊区的天气适不适合出行，适合的话帮我在"
            "日历上记一个周六上午九点到下午四点的『班级徒步』活动。"
        ),
        "expect": [("weather", "forecast"), ("calendar", "create_event")],
    },
    {
        "id": "btc_report",
        "prompt": (
            "帮我做一份比特币行情小报告：查一下 BTC 的实时价格，取最近 30 天的日 K 线"
            "数据画一张走势图，再把这些市值换算成人民币总额。"
        ),
        "expect": [("crypto", "price"), ("crypto", "kline"), ("chart", "render_chart"), ("forex", "convert")],
    },
]


def get_tasks() -> list[dict]:
    """返回任务列表（深拷贝防御调用方改写），expected_tools 已换算成 wrapped 名。"""
    tasks: list[dict] = []
    for t in _TASK_DEFS:
        tasks.append({
            "id": t["id"],
            "prompt": t["prompt"],
            "expected_tools": [wrapped_tool_name(s, n) for s, n in t["expect"]],
        })
    return tasks


__all__ = ["get_tasks"]
