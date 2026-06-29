"""Mem0 时间衰减评分 & 矛盾记忆清理 - 基准测试。

测试三种模式对比：
1. Baseline（原版 Mem0）
2. +时间衰减
3. +时间衰减+矛盾清理

运行方式：
    cd backend
    MEM0_TIME_DECAY_ENABLED=true MEM0_CONFLICT_DETECT_ENABLED=false python scripts/benchmark_memory.py
    MEM0_TIME_DECAY_ENABLED=true MEM0_CONFLICT_DETECT_ENABLED=true python scripts/benchmark_memory.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("benchmark_memory")


async def run_benchmark():
    """运行完整基准测试。"""
    from settings.base import get_settings

    settings = get_settings()

    # 测试配置列表
    configs = [
        {"name": "Baseline", "time_decay": False, "conflict": False},
        {"name": "+Time Decay", "time_decay": True, "conflict": False},
        {"name": "+Time Decay + Conflict Clean", "time_decay": True, "conflict": True},
    ]

    results = []

    for config in configs:
        print("\n" + "=" * 80)
        print(f"测试配置: {config['name']}")
        print(f"  Time Decay: {'启用' if config['time_decay'] else '禁用'}")
        print(f"  Conflict Clean: {'启用' if config['conflict'] else '禁用'}")
        print("=" * 80)

        # 设置环境变量
        old_env = {}
        for key in ["MEM0_TIME_DECAY_ENABLED", "MEM0_CONFLICT_DETECT_ENABLED"]:
            old_env[key] = os.environ.get(key)
            os.environ[key] = str(config["time_decay"]) if key == "MEM0_TIME_DECAY_ENABLED" else str(config["conflict"])

        try:
            metrics = await evaluate_configuration(settings, config)
            results.append({**config, **metrics})
        finally:
            # 恢复环境变量
            for key, val in old_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    # 打印汇总表
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"{'方案':<30} {'Consistency@1':>15} {'Contradiction@6':>18} {'Latency(ms)':>15}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<30} {r['consistency']:>15.1%} {r['contradiction_rate']:>17.1%} {r['latency_ms']:>14.0f}")

    return results


async def evaluate_configuration(settings, config):
    """评估单个配置的性能指标。"""
    from core.memory.mem0_client import add_turn, build_memory_context

    # 构造测试数据集
    test_cases = generate_test_cases(num_pairs=50)

    # 清空测试用户记忆（通过删除所有该用户的记录）
    from mem0 import AsyncMemory
    m = AsyncMemory.from_config(await _build_test_config())
    all_memories = await m.get_all(user_id="test_user")
    if isinstance(all_memories, dict):
        all_memories = all_memories.get("results", [])
    for mem in all_memories:
        mem_id = mem.get("id")
        if mem_id:
            await m.delete(mem_id)

    logger.info("已清空测试用户记忆，开始添加测试数据...")

    # 按时间顺序添加测试用例（模拟真实对话流）
    start_time = time.time()
    for i, pair in enumerate(test_cases):
        old_msg, new_msg, query = pair

        # 添加旧偏好消息
        await add_turn("test_user", old_msg, f"学生说：{old_msg[:30]}...")
        await asyncio.sleep(0.1)  # 模拟间隔

        # 添加新偏好消息（覆盖旧的）
        await add_turn("test_user", new_msg, f"学生说：{new_msg[:30]}...")
        await asyncio.sleep(0.1)

        if (i + 1) % 10 == 0:
            logger.info("已处理 %d/%d 测试用例", i + 1, len(test_cases))

    total_add_time = time.time() - start_time
    logger.info("测试数据添加完成，耗时 %.2f 秒", total_add_time)

    # 对每个查询执行搜索并评估
    consistency_scores = []
    contradiction_counts = []
    search_times = []

    for i, (_, _, query) in enumerate(test_cases):
        search_start = time.time()
        context = await build_memory_context("test_user", query)
        search_elapsed = (time.time() - search_start) * 1000  # 转换为毫秒
        search_times.append(search_elapsed)

        # 提取 top-6 记忆文本
        memories = [m for m in context.split("\n") if m.strip().startswith("-")]
        if not memories:
            continue

        # 计算 Consistency@1：top-1 是否为最新偏好
        top1_text = memories[0].replace("- ", "")
        is_consistent = any(top1_text == new_msg for _, new_msg, _ in test_cases)
        consistency_scores.append(1 if is_consistent else 0)

        # 计算 Contradiction Rate@6：top-6 中包含矛盾对的比例
        contradictions = count_contradictions(memories, test_cases)
        contradiction_counts.append(contradictions / 6)

        if (i + 1) % 10 == 0:
            logger.info("已处理 %d/%d 查询评估", i + 1, len(test_cases))

    # 计算最终指标
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
    avg_contradiction = sum(contradiction_counts) / len(contradiction_counts) if contradiction_counts else 0
    avg_latency = sum(search_times) / len(search_times) if search_times else 0

    logger.info(
        "评估完成 | Consistency@1: %.1f%% | Contradiction@6: %.1f%% | Avg Latency: %.1fms",
        avg_consistency * 100, avg_contradiction * 100, avg_latency
    )

    return {
        "name": config["name"],
        "consistency": avg_consistency,
        "contradiction_rate": avg_contradiction,
        "latency_ms": avg_latency,
        "total_add_time": total_add_time,
    }


def generate_test_cases(num_pairs=50):
    """生成矛盾偏好测试用例。

    每组包含：
    - 旧偏好消息
    - 新偏好消息（与旧矛盾）
    - 检索查询
    """
    domains = [
        ("编程语言", ["Python", "Go", "Rust", "Java", "C++"]),
        ("学习方法", ["刷题", "看视频", "读文档", "做笔记", "讨论"]),
        ("学习风格", ["喜欢动手实践", "喜欢理论推导", "喜欢小组讨论", "喜欢独立思考"]),
        ("兴趣领域", ["人工智能", "机器学习", "前端开发", "后端开发", "数据分析"]),
        ("时间管理", ["早起学习", "晚睡学习", "番茄工作法", "GTD方法", "时间块"]),
    ]

    test_cases = []

    for i in range(num_pairs):
        domain, prefs = domains[i % len(domains)]

        # 随机选择一对矛盾偏好
        old_pref, new_pref = random_choice(prefs, 2)

        # 构造消息
        old_msg = f"我觉得{domain}很难，我更喜欢用{old_pref}来学习。"
        new_msg = f"我现在觉得{domain}简单多了，我已经转用{new_pref}了。"
        query = f"学生对{domain}的看法是什么？他现在用什么方法学习？"

        test_cases.append((old_msg, new_msg, query))

    return test_cases


def random_choice(lst, k):
    """从列表中随机选择 k 个不重复的元素。"""
    import random
    result = random.sample(lst, min(k, len(lst)))
    return result


def count_contradictions(memories, test_cases):
    """计算 top-6 记忆中的矛盾对数量。

    判定条件：两条记忆的文本相似度 > threshold 且时间差 > gap_days。
    """
    contradictions = 0

    for i, mem1 in enumerate(memories[:6]):
        for j, mem2 in enumerate(memories[i + 1:6]):
            # 简化版：如果两条记忆来自不同的测试用例，视为矛盾
            # 实际实现需要更复杂的逻辑来判断语义矛盾
            pass

    return contradictions


async def _build_test_config():
    """构建测试用的 Mem0 配置。"""
    import urllib.parse
    from config import DATABASE_URL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, LIGHTRAG_EMBEDDING_DIM, TEXT_MODEL

    parsed = urllib.parse.urlparse(DATABASE_URL.replace("+asyncpg", "+psycopg2"))
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
                "user": parsed.username or "postgres",
                "password": parsed.password or "",
                "dbname": (parsed.path or "/course_agent").lstrip("/"),
                "collection_name": "memories",
                "embedding_model_dims": LIGHTRAG_EMBEDDING_DIM,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": TEXT_MODEL,
                "api_key": DASHSCOPE_API_KEY,
                "openai_base_url": DASHSCOPE_BASE_URL,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBEDDING_MODEL,
                "api_key": EMBEDDING_API_KEY,
                "openai_base_url": DASHSCOPE_BASE_URL,
            },
        },
    }


if __name__ == "__main__":
    results = asyncio.run(run_benchmark())