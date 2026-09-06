"""合成 MCP 工具池：10 个虚构 server × 8 工具 = 80 个 MCPToolAdapter。

本项目 data/mcp.json 只有个位数工具，"要不要主动发现"在这个 n 下不成立；本模块
用**生产数据结构** MCPToolAdapter（core/mcp/adapter.py）把池子造大，跨域覆盖思路
对齐 chapter4 tools_library.py（arxiv/weather/forex/...），但 schema 走本项目
wrapped_name（mcp_<server>_<tool>）+ to_openai_schema，不另造格式。

每个工具配一个确定性 mock 执行器（返回一句示例文本），注册进生产 ToolRegistry，
执行路由复用 execute_tool，不重写分发。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from core.agent.registry import ToolEntry, ToolRegistry, get_tool_registry
from core.agent.tool_protocol import ToolResult
from core.mcp.adapter import MCPToolAdapter

# 每工具 1-3 个参数：(参数名, 类型, 中文描述)。类型只用于拼 JSON schema。
Spec = list[tuple[str, str, str]]
ToolDef = tuple[str, str, Spec]  # (original_name, description, params)

_SERVERS: dict[str, list[ToolDef]] = {
    "arxiv": [
        ("search", "在 arXiv 学术论文库按关键词检索论文，返回标题、作者与摘要", [("query", "string", "检索关键词"), ("max_results", "integer", "返回篇数上限")]),
        ("get_abstract", "获取某篇论文的完整摘要与发表信息", [("paper_id", "string", "论文 arXiv 编号")]),
        ("download_pdf", "下载论文 PDF 全文到本地", [("paper_id", "string", "论文 arXiv 编号"), ("path", "string", "保存路径")]),
        ("cite_bibtex", "把论文引用信息导出为 BibTeX 格式", [("paper_id", "string", "论文 arXiv 编号")]),
        ("related_papers", "查找与某篇论文主题相关的其它论文", [("paper_id", "string", "论文 arXiv 编号"), ("max_results", "integer", "返回篇数上限")]),
        ("author_profile", "查询某位学者的发文历史、领域与引用统计", [("author", "string", "学者姓名")]),
        ("trend_analysis", "统计某研究主题近年的论文数量变化趋势", [("topic", "string", "研究主题"), ("years", "integer", "统计年数")]),
        ("fulltext_search", "在论文全文中检索包含指定短语的段落", [("phrase", "string", "要检索的短语"), ("max_results", "integer", "返回段数上限")]),
    ],
    "weather": [
        ("current", "查询某城市当前的实时天气（温度、湿度、天气现象）", [("location", "string", "城市名")]),
        ("forecast", "查询某城市未来若干天的天气预报", [("location", "string", "城市名"), ("days", "integer", "预报天数")]),
        ("historical", "查询某城市过去某日期的历史天气记录", [("location", "string", "城市名"), ("date", "string", "日期 YYYY-MM-DD")]),
        ("air_quality", "查询某城市的空气质量指数 AQI 与污染物浓度", [("location", "string", "城市名")]),
        ("clothing_advice", "根据天气给出穿衣出行建议", [("location", "string", "城市名")]),
        ("uv_index", "查询某城市的紫外线指数与防晒建议", [("location", "string", "城市名")]),
        ("city_compare", "对比多个城市同一天的天气差异", [("locations", "string", "城市名列表，逗号分隔")]),
        ("severe_alert", "查询某城市生效中的灾害天气预警", [("location", "string", "城市名")]),
    ],
    "forex": [
        ("get_rate", "查询两种法定货币之间的实时外汇汇率", [("base", "string", "基准货币代码"), ("quote", "string", "目标货币代码")]),
        ("convert", "把一笔金额从一种法定货币换算成另一种", [("amount", "number", "金额"), ("base", "string", "基准货币代码"), ("quote", "string", "目标货币代码")]),
        ("historical_chart", "查询货币对的历史汇率走势数据", [("base", "string", "基准货币代码"), ("quote", "string", "目标货币代码"), ("days", "integer", "回看天数")]),
        ("cross_table", "生成多种货币两两之间的交叉汇率表", [("currencies", "string", "货币代码列表，逗号分隔")]),
        ("rate_alert", "设置汇率提醒：到达目标价时通知", [("base", "string", "基准货币代码"), ("quote", "string", "目标货币代码"), ("target", "number", "目标汇率")]),
        ("major_pairs", "列出美元、欧元、日元等主要货币对的最新汇率", []),
        ("weekly_outlook", "给出某货币对未来一周的汇率走势分析与预测", [("base", "string", "基准货币代码"), ("quote", "string", "目标货币代码")]),
        ("central_bank_policy", "查询各国央行最新利率决议与货币政策动向", [("bank", "string", "央行名称")]),
    ],
    "github": [
        ("repo_stats", "查询某 GitHub 仓库的 star、fork、issue 数等基本统计", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名")]),
        ("list_contributors", "获取某仓库的贡献者提交统计排行", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名")]),
        ("issue_search", "按关键词检索某仓库的 issue 与讨论", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名"), ("query", "string", "检索关键词")]),
        ("commit_diff", "查看某次提交的代码变更 diff", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名"), ("sha", "string", "提交哈希")]),
        ("pr_review", "获取某 pull request 的评审意见与结论", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名"), ("pr_number", "integer", "PR 编号")]),
        ("star_history", "查询某仓库的历史 star 增长曲线数据", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名")]),
        ("code_search", "在 GitHub 全站按代码片段检索仓库文件", [("query", "string", "代码检索表达式")]),
        ("release_notes", "查询某仓库最新 release 的更新日志", [("owner", "string", "仓库所有者"), ("repo", "string", "仓库名")]),
    ],
    "calendar": [
        ("create_event", "在日历上创建一个新的事件或日程", [("title", "string", "事件标题"), ("start", "string", "开始时间"), ("end", "string", "结束时间")]),
        ("list_events", "列出某日期范围内的已有日程", [("start", "string", "起始日期"), ("end", "string", "结束日期")]),
        ("delete_event", "删除一个已有的日历事件", [("event_id", "string", "事件 id")]),
        ("find_slots", "在日程表中寻找最近的空闲时间段", [("duration_min", "integer", "需要的时长（分钟）"), ("date", "string", "日期")]),
        ("reschedule", "把一个已有事件改期到新时间", [("event_id", "string", "事件 id"), ("start", "string", "新的开始时间")]),
        ("agenda_daily", "生成某天的日程摘要（按时间线）", [("date", "string", "日期")]),
        ("week_overview", "生成本周日程总览与空闲率统计", []),
        ("conflict_check", "检查某时间段与已有日程是否冲突", [("start", "string", "开始时间"), ("end", "string", "结束时间")]),
    ],
    "news": [
        ("search", "按关键词检索相关的最新新闻报道", [("query", "string", "检索关键词"), ("max_results", "integer", "返回条数上限")]),
        ("trending", "获取当前热门话题榜单", [("category", "string", "话题类别")]),
        ("category_brief", "获取某领域（科技/财经/教育等）的今日要闻简报", [("category", "string", "领域类别")]),
        ("sentiment_analysis", "分析某话题下新闻舆论的情感倾向（正面/负面占比）", [("topic", "string", "话题关键词")]),
        ("fact_check", "对一条说法核查多方信源并给出可信度评级", [("claim", "string", "要核查的说法")]),
        ("timeline_track", "梳理某事件的时间线与关键节点报道", [("event", "string", "事件关键词")]),
        ("source_compare", "对比不同媒体对同一事件的报道立场差异", [("event", "string", "事件关键词")]),
        ("keyword_alert", "订阅关键词：相关新闻出现时提醒", [("keyword", "string", "订阅关键词")]),
    ],
    "translate": [
        ("text", "把一段文本从一种语言翻译成另一种语言", [("text", "string", "要翻译的文本"), ("source", "string", "源语言"), ("target", "string", "目标语言")]),
        ("document", "翻译整份文档并保留排版", [("file_path", "string", "文档路径"), ("target", "string", "目标语言")]),
        ("glossary_lookup", "查询专业术语在指定领域的标准译法", [("term", "string", "术语"), ("domain", "string", "领域")]),
        ("detect_language", "检测一段文本的语言种类", [("text", "string", "要检测的文本")]),
        ("tone_rewrite", "把文本改写成指定语气风格（正式/口语/学术）", [("text", "string", "要改写的文本"), ("tone", "string", "目标语气")]),
        ("back_translate", "回译校验：译文再译回源语言以检查语义偏差", [("text", "string", "要校验的译文"), ("source", "string", "源语言")]),
        ("term_extract", "从文本中抽取关键术语并给出双语对照表", [("text", "string", "要抽取的文本")]),
        ("subtitle", "为视频字幕文件生成翻译版本", [("file_path", "string", "字幕文件路径"), ("target", "string", "目标语言")]),
    ],
    "chart": [
        ("render_chart", "根据数据渲染柱状图/折线图/饼图等可视化图表", [("data", "string", "数据（JSON 数组）"), ("chart_type", "string", "图表类型"), ("title", "string", "图表标题")]),
        ("data_table", "把数据整理成 Markdown 表格", [("data", "string", "数据（JSON 数组）"), ("columns", "string", "要展示的列")]),
        ("export_png", "把已生成的图表导出为 PNG 图片文件", [("chart_id", "string", "图表 id"), ("path", "string", "导出路径")]),
        ("style_apply", "给图表应用配色与主题样式", [("chart_id", "string", "图表 id"), ("theme", "string", "主题名")]),
        ("axis_config", "配置图表坐标轴的刻度、范围与标签", [("chart_id", "string", "图表 id"), ("axis", "string", "轴名"), ("config", "string", "轴配置 JSON")]),
        ("embed_html", "生成可嵌入网页的交互式图表代码", [("chart_id", "string", "图表 id")]),
        ("compare_charts", "把多张图表合并为对比子图布局", [("chart_ids", "string", "图表 id 列表，逗号分隔")]),
        ("heatmap", "根据矩阵数据渲染热力图", [("data", "string", "矩阵数据（JSON）"), ("title", "string", "图表标题")]),
    ],
    "crypto": [
        ("price", "查询比特币、以太坊等加密货币的实时价格与涨跌幅", [("symbol", "string", "币种符号，如 BTC")]),
        ("kline", "查询加密货币的 K 线历史行情数据", [("symbol", "string", "币种符号"), ("interval", "string", "K 线周期"), ("limit", "integer", "返回根数")]),
        ("market_depth", "查询交易盘口的买卖挂单深度", [("symbol", "string", "币种符号")]),
        ("whale_alert", "监控大额加密货币转账的巨鲸动向", [("symbol", "string", "币种符号"), ("min_amount", "number", "最小金额")]),
        ("fear_greed", "查询加密市场的恐慌与贪婪指数", []),
        ("portfolio_value", "计算一组持仓币种的组合总市值", [("holdings", "string", "持仓明细（JSON）")]),
        ("gas_fee", "查询区块链当前的网络手续费（gas）水平", [("chain", "string", "链名")]),
        ("staking_yield", "查询某币种质押（staking）的年化收益率", [("symbol", "string", "币种符号")]),
    ],
    "file": [
        ("download", "从 URL 下载文件保存到本地", [("url", "string", "文件地址"), ("path", "string", "保存路径")]),
        ("upload", "把本地文件上传到云存储并返回分享链接", [("path", "string", "本地文件路径")]),
        ("list_dir", "列出目录下的文件与子目录", [("path", "string", "目录路径")]),
        ("search_files", "按文件名模式在目录树中查找文件", [("pattern", "string", "文件名通配模式"), ("root", "string", "查找根目录")]),
        ("compress", "把文件或目录压缩成 zip 包", [("path", "string", "要压缩的路径"), ("archive", "string", "压缩包输出路径")]),
        ("diff_files", "对比两个文件的差异并生成 diff 报告", [("path_a", "string", "文件 A"), ("path_b", "string", "文件 B")]),
        ("convert_format", "转换文件格式（如 docx→pdf、csv→xlsx）", [("path", "string", "源文件路径"), ("target_format", "string", "目标格式")]),
        ("checksum", "计算文件的校验和（md5/sha256）验证完整性", [("path", "string", "文件路径"), ("algo", "string", "算法名")]),
    ],
}

TOOL_TIMEOUT = 30  # mock 无所谓，沿用生产的常规量级


def build_pool(size: int | None = None) -> list[MCPToolAdapter]:
    """全量构造 80 个 MCPToolAdapter（deferred=True，与生产 MCP 工具默认一致）。

    size 仅用于 smoke 裁剪（取前 size 个）；正式对比跑全量。
    """
    adapters: list[MCPToolAdapter] = []
    for server, tools in _SERVERS.items():
        for original_name, description, params in tools:
            properties: dict[str, Any] = {
                name: {"type": ptype, "description": pdesc}
                for name, ptype, pdesc in params
            }
            required = [name for name, _, _ in params[:1]] or []
            adapters.append(MCPToolAdapter(
                server_name=server,
                original_name=original_name,
                description=description,
                input_schema={"type": "object", "properties": properties, "required": required},
                tool_timeout=TOOL_TIMEOUT,
                deferred=True,
            ))
    if size is not None:
        adapters = adapters[:size]
    return adapters


def pool_to_dict(pool: list[MCPToolAdapter]) -> dict[str, MCPToolAdapter]:
    """wrapped_name → adapter（DeferredToolLoader 的 pool 入参格式）。"""
    return {a.wrapped_name: a for a in pool}


def make_mock_executor(adapter: MCPToolAdapter) -> Callable[..., Any]:
    """确定性 mock 执行器：与生产 ToolExecutor 同签名（关键字 course_id/user_id + kwargs）。"""

    async def _execute(*, course_id: str = "", user_id: str = "", **kwargs: Any) -> ToolResult:
        args_preview = json.dumps(kwargs, ensure_ascii=False)[:200]
        return ToolResult(
            content=(
                f"[mock] {adapter.wrapped_name} 已执行，参数：{args_preview}。"
                f"返回示例数据：这是 {adapter.server_name}/{adapter.original_name} 的模拟结果，"
                "内容已按任务要求生成。"
            ),
            success=True,
        )

    return _execute


def register_mock_tools(registry: ToolRegistry | None = None, pool: list[MCPToolAdapter] | None = None) -> list[str]:
    """把合成工具注册进 ToolRegistry（幂等：同名覆盖），返回注册名列表。"""
    reg = registry if registry is not None else get_tool_registry()
    tools = pool if pool is not None else build_pool()
    names: list[str] = []
    for adapter in tools:
        reg.register(ToolEntry(
            name=adapter.wrapped_name,
            schema=adapter.to_openai_schema(),
            executor=make_mock_executor(adapter),
            deferred=True,
        ))
        names.append(adapter.wrapped_name)
    return names


def unregister_mock_tools(registry: ToolRegistry | None = None, names: list[str] | None = None) -> None:
    """清理注册（测试隔离用；names 为 None 时按全池反查）。"""
    reg = registry if registry is not None else get_tool_registry()
    if names is None:
        names = [a.wrapped_name for a in build_pool()]
    for name in names:
        reg.unregister(name)


__all__ = [
    "build_pool",
    "pool_to_dict",
    "make_mock_executor",
    "register_mock_tools",
    "unregister_mock_tools",
]
