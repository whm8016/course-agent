"""MCP 主动发现 vs deferred 清单对比 demo 评测包。

两臂共用生产 run_agent_loop 调度内核，只换"工具怎么露出来"这一层：
- deferred_exact（现状）：manifest 全量索引进 system prompt + load_tools 精确名加载
- active_discovery（新臂）：零索引，仅一个 discover_tools 语义检索元工具
"""
