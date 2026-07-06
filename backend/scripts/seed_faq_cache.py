"""
预填充电路分析知识库的 FAQ 高频问题缓存。

脚本会遍历预定义的 150 道电路分析高频问题，对每个问题调用 LLM 生成答案，
然后写入 Redis 缓存（faq:answer:{course_id}:{hash} + faq:count:{course_id}），
使学生常见问题可以直接命中缓存秒回。

Usage:
    # 预览模式（不写入 Redis）
    python scripts/seed_faq_cache.py --dry-run

    # 实际写入
    python scripts/seed_faq_cache.py

    # 指定课程 ID 和并发数
    python scripts/seed_faq_cache.py --course-id mycourse --concurrency 5

    # 只处理前 20 道题
    python scripts/seed_faq_cache.py --count 20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 确保 backend/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import get_settings
FAQ_CACHE_THRESHOLD = get_settings().question.faq_cache_threshold
REDIS_URL = get_settings().db.redis_url.get_secret_value()
from core.llm.llm import chat_complete
import hashlib
import redis.asyncio as aioredis

# ---------------------------------------------------------------------------
# 系统提示词：引导 LLM 生成适合知识库的电路分析答案
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一位《电路分析基础实验》课程的助教。请根据学生的提问，给出准确、简洁、
易懂的回答。回答应包含：
1. 核心概念的解释
2. 关键公式（如有）
3. 实验相关注意事项（如适用）

回答控制在 150-300 字，语言精练，避免冗余。不要加 "同学你好" 之类的开场白，
直接回答问题。"""

# ---------------------------------------------------------------------------
# 150 道高频问题（按实验主题分类）
# ---------------------------------------------------------------------------

FAQ_QUESTIONS: list[str] = [
    # ── 一、常用电子仪器原理与使用（15题）──
    "示波器的基本工作原理是什么？",
    "如何用示波器测量信号的频率？",
    "如何用示波器测量信号的幅值和峰峰值？",
    "示波器的触发电平（LEVEL）旋钮有什么作用？",
    "示波器的垂直电压标尺（V/div）怎么设置？",
    "示波器的水平时基标尺（s/div）怎么调节？",
    "信号发生器可以输出哪几种波形？",
    "如何用信号发生器产生指定频率和幅值的正弦波？",
    "数字万用表可以测量哪些参数？",
    "用万用表测量电阻时需要注意什么？",
    "用万用表测量直流电压和交流电压有什么区别？",
    "用万用表如何判断二极管的好坏？",
    "直流稳压电源的使用注意事项有哪些？",
    "示波器的双踪模式怎么使用？",
    "如何用示波器的光标测量功能测量相位差？",

    # ── 二、元件的伏安特性测试（15题）──
    "什么是伏安特性曲线？",
    "线性电阻的伏安特性曲线是什么样的？",
    "非线性电阻和线性电阻的伏安特性有什么区别？",
    "半导体二极管的伏安特性曲线有什么特点？",
    "什么是二极管的正向导通电压？一般是多少？",
    "二极管的正向特性和反向特性分别是什么？",
    "理想电压源的伏安特性曲线是什么样的？",
    "实际电压源和理想电压源有什么区别？",
    "测量伏安特性时电压表内阻对结果有什么影响？",
    "测量伏安特性时电流表内阻对结果有什么影响？",
    "电压表接在1-1'和2-2'位置有什么区别？怎么选择？",
    "什么是电压源的内阻？如何测量？",
    "如何用实验方法测定实际电压源的伏安特性？",
    "限流电阻在二极管伏安特性测试中起什么作用？",
    "什么是元件的双向性？哪些元件具有双向性？",

    # ── 三、电阻串并联电路与基尔霍夫定律（15题）──
    "电阻串联电路的特点是什么？",
    "电阻并联电路的特点是什么？",
    "如何计算串联电路的总电阻？",
    "如何计算并联电路的总电阻？",
    "串联电路中电压是如何分配的？",
    "并联电路中电流是如何分配的？",
    "基尔霍夫电流定律（KCL）的内容是什么？",
    "基尔霍夫电压定律（KVL）的内容是什么？",
    "如何用实验验证基尔霍夫电流定律？",
    "如何用实验验证基尔霍夫电压定律？",
    "什么是支路电流的参考方向？",
    "电流表和电压表在电路中分别怎么连接？",
    "使用直流电表时为什么要关注极性？",
    "什么是相对误差？如何计算？",
    "验证基尔霍夫定律时误差来源有哪些？",

    # ── 四、叠加定理（12题）──
    "叠加定理的内容是什么？",
    "叠加定理适用于什么类型的电路？",
    '叠加定理中"一个电源单独作用"是什么意思？',
    "理想电压源不作用时如何处理？",
    "理想电流源不作用时如何处理？",
    "叠加定理能否用于计算功率？为什么？",
    "如何用实验验证叠加定理？",
    "叠加定理实验中为什么要保留电源内阻？",
    "叠加定理实验中电流表指针反偏说明什么？",
    "为什么要用万用表重新测定电源输出电压？",
    "叠加定理的适用条件和局限性是什么？",
    "用具体数值说明为什么功率不能用叠加定理计算？",

    # ── 五、戴维南定理（含源一端口网络）（15题）──
    "戴维南定理的内容是什么？",
    "什么是含源一端口网络？",
    "如何确定戴维南等效电路的开路电压？",
    "如何确定戴维南等效电路的等效内阻？",
    "测量开路电压有哪几种方法？",
    "什么是补偿法测开路电压？有什么优点？",
    "测量等效内阻有哪几种方法？",
    "什么是两次电压测量法？有什么优点？",
    "短路电流法测量等效电阻有什么局限性？",
    "如何用实验验证戴维南定理？",
    "什么是有源二端网络的外特性？",
    "戴维南等效电路的外特性与原网络有什么关系？",
    "什么是诺顿定理？与戴维南定理的关系是什么？",
    "实验中如何避免电表内阻对测量的影响？",
    "如何画出有源二端网络的外特性曲线？",

    # ── 六、三端变阻器（分压器）（10题）──
    "三端变阻器有哪两种接法？分别用于什么场合？",
    "什么是分压器的调压特性？",
    "分压器的调压特性曲线在什么条件下接近直线？",
    "负载电阻对分压器调压特性有什么影响？",
    "选择分压器变阻器阻值时需要考虑哪些因素？",
    "R0太小对分压器有什么坏处？",
    "空载时分压器的输出电压怎么计算？",
    "什么是参变量？本实验中哪个量是参变量？",
    "如何选择分压变阻器的额定功率？",
    "分压器对电源多取的电流与什么有关？",

    # ── 七、电路过渡过程（RC/RLC暂态）（15题）──
    "什么是电路的过渡过程？",
    "RC微分电路的工作原理是什么？",
    "RC积分电路的工作原理是什么？",
    "微分电路和积分电路有什么区别？",
    "什么是电路的时间常数？",
    "时间常数对电路过渡过程有什么影响？",
    "如何用示波器观察RC电路的过渡过程？",
    "为什么用矩形波代替直流电压来研究过渡过程？",
    "RLC串联电路的过渡过程有哪几种情况？",
    "什么是过阻尼、临界阻尼和欠阻尼？",
    "RLC电路在什么条件下会产生衰减振荡？",
    "什么是无阻尼振荡频率？",
    "RLC并联电路的过渡过程有什么特点？",
    "微分电路中时间常数很小时输出电压与输入电压有什么关系？",
    "积分电路中时间常数很大时输出电压与输入电压有什么关系？",

    # ── 八、交流电路实验（R/L/C阻抗）（15题）──
    "电阻元件在交流电路中的阻抗特性是什么？",
    "电阻元件的电压和电流相位关系是什么？",
    "什么是感抗？如何计算？",
    "电感元件的电压和电流相位关系是什么？",
    "电感元件的阻抗与频率有什么关系？",
    "什么是容抗？如何计算？",
    "电容元件的电压和电流相位关系是什么？",
    "电容元件的阻抗与频率有什么关系？",
    "如何用伏安法测定电感的电感量？",
    "如何用伏安法测定电容的电容量？",
    "交流电路中如何验证基尔霍夫电流定律？",
    "电感本身有电阻时对测量结果有什么影响？",
    "如何测量电感或电容元件的电流？",
    "R、L、C元件在交直流电路中的性能有什么不同？",
    "交流毫伏表的使用方法是什么？",

    # ── 九、二端口网络（12题）──
    "什么是二端口网络？",
    "二端口网络的传输参数（A参数）有哪些？",
    "如何用实验方法测定二端口网络的A参数？",
    "什么是开路阻抗和短路阻抗？",
    "互易二端口网络的参数有什么约束条件？",
    "对称二端口网络的参数有什么特点？",
    "如何用T型等效电路替代二端口网络？",
    "T型等效电路的参数怎么由A参数计算？",
    "二端口网络的π型等效电路是什么样的？",
    "如何验证二端口网络等效电路的正确性？",
    "二端口网络参数与外部激励有什么关系？",
    "在输入端和输出端分别测量的方法有什么优势？",

    # ── 十、RC网络频率特性（15题）──
    "什么是电路的频率特性？",
    "RC串并联选频电路（文氏电路）的结构是什么？",
    "文氏电路在低频和高频时的等效电路分别是什么？",
    "文氏电路的谐振频率怎么计算？",
    "文氏电路在谐振频率时输出电压和输入电压的比值是多少？",
    "什么是网络函数（传递函数）？",
    "什么是幅频特性和相频特性？",
    "如何用示波器观察李萨育图形来测定谐振频率？",
    "双T网络的频率特性与文氏电路有什么不同？",
    "双T网络的截止频率是什么意思？",
    "如何利用文氏电路组成正弦波振荡器？",
    "正弦波振荡器的起振条件是什么？",
    "RC正弦波振荡器中放大器的放大倍数应该多大？",
    "如何测定文氏电路的幅频特性和相频特性？",
    "文氏电路选频的原理是什么？",

    # ── 十一、滤波器实验（11题）──
    "滤波器的作用是什么？",
    "如何设计一个滤除特定频率谐波的滤波器？",
    "并联谐振在滤波器中起什么作用？",
    "串联谐振在滤波器中起什么作用？",
    "如何用实验方法确定滤波器的电感和电容参数？",
    "滤波器对基波和高次谐波分别呈现什么阻抗特性？",
    "非正弦电源的波形主要由哪些频率成分组成？",
    "如果非正弦电源含有直流分量，滤波器还能正常工作吗？",
    "如果还有五次七次等高次谐波，滤波器还能完成滤波任务吗？",
    "如何用示波器观察滤波前后的波形变化？",
    "什么是谐振滤波器的设计步骤？",
]

# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

# 脚本自有的 Redis 连接池（不使用 cache.py 的 _get_pool，因为它的
# socket_connect_timeout=2 在某些环境下会导致连接超时）
_redis_pool: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


def _faq_hash(question: str) -> str:
    """与 cache.py 中 _faq_hash 保持一致。"""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:16]


async def _faq_answer_set(course_id: str, question: str, answer: str) -> None:
    r = _get_redis()
    key = f"faq:answer:{course_id}:{_faq_hash(question)}"
    await r.set(key, answer)


async def _faq_record(course_id: str, question: str) -> int:
    r = _get_redis()
    key = f"faq:count:{course_id}"
    count = await r.zincrby(key, 1, question.strip())
    return int(count)

# 将问题计数直接设为 threshold，使其达到缓存命中条件
_SEED_COUNT = max(FAQ_CACHE_THRESHOLD, 3)


async def _generate_and_seed(
    question: str,
    course_id: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
) -> dict:
    """对单个问题：调 LLM 生成答案 → 写入 Redis。返回统计 dict。"""
    async with semaphore:
        t0 = time.perf_counter()

        # 1. 调用 LLM 生成答案
        try:
            answer = await chat_complete(
                system_prompt=SYSTEM_PROMPT,
                history=[],
                user_message=question,
                temperature=0.3,  # 低温度，答案更稳定
                max_tokens=512,
            )
        except Exception as e:
            return {"question": question, "ok": False, "error": str(e)}

        elapsed = time.perf_counter() - t0

        if dry_run:
            return {"question": question, "ok": True, "answer_len": len(answer), "elapsed": elapsed, "dry_run": True}

        # 2. 写入缓存答案
        await _faq_answer_set(course_id, question, answer)

        # 3. 设置问题计数（多次调用 _faq_record 使其 >= threshold）
        for _ in range(_SEED_COUNT):
            await _faq_record(course_id, question)

        return {"question": question, "ok": True, "answer_len": len(answer), "elapsed": elapsed}


async def run(args: argparse.Namespace) -> None:
    questions = FAQ_QUESTIONS[: args.count] if args.count else FAQ_QUESTIONS
    total = len(questions)

    print(f"\n{'='*60}")
    print("  电路分析 FAQ 预缓存脚本")
    print(f"  课程 ID: {args.course_id}")
    print(f"  问题数量: {total}")
    print(f"  并发数: {args.concurrency}")
    print(f"  缓存阈值: {FAQ_CACHE_THRESHOLD}")
    print(f"  模式: {'预览 (dry-run)' if args.dry_run else '实际写入'}")
    print(f"  Redis: {REDIS_URL}")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [_generate_and_seed(q, args.course_id, semaphore, args.dry_run) for q in questions]

    success = 0
    failed = 0
    t_start = time.perf_counter()

    # 使用 as_completed 打印进度
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result["ok"]:
            success += 1
            q_short = result["question"][:30]
            extra = ""
            if "answer_len" in result:
                extra = f" ({result['answer_len']}字, {result['elapsed']:.1f}s)"
            if result.get("dry_run"):
                extra += " [dry-run]"
            print(f"  [OK] [{success+failed}/{total}] {q_short}...{extra}", flush=True)
        else:
            failed += 1
            print(f"  [FAIL] [{success+failed}/{total}] {result['question'][:30]}... 错误: {result['error'][:80]}", flush=True)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"  完成！成功: {success}  失败: {failed}  耗时: {elapsed:.1f}s")
    if args.dry_run:
        print("  （预览模式，未写入 Redis）")
    else:
        print(f"  已写入课程 {args.course_id} 的 FAQ 缓存")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="预填充电路分析 FAQ 缓存")
    parser.add_argument("--course-id", default="mycourse", help="课程 ID（默认 mycourse）")
    parser.add_argument("--count", type=int, default=0, help="只处理前 N 道题（默认全部）")
    parser.add_argument("--concurrency", type=int, default=5, help="LLM 并发数（默认 5）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入 Redis")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
