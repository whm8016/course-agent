"""
模拟 question.py：一个 asyncio.Queue，一个 log_pusher 慢速发消息。
看两种「客户端行为」在打印上的差异（无需真 WebSocket）。
"""
import asyncio
import time

# 发一条上网的延迟（毫秒级），让队列里容易堆还没发出的消息
SEND_DELAY_SEC = 0.05

async def slow_client_receive(msg, mode: str):
    """mode: 'bad' 在 progress+complete 时认为结束; 'good' 等到 type=complete 才结束"""
    t = msg.get("type")
    if t == "result":
        print(f"  客户端收到: result id={msg.get('id')}")
    elif t == "progress" and msg.get("stage") == "complete":
        print(f"  客户端收到: progress stage=complete（只是进度，不是最终关门信号）")
        if mode == "bad":
            print("  [bad] 这里就当作「全结束了」→ 下面不再接收，模拟关连接\n")
            return "STOP"  # 模拟关 WebSocket，后面消息丢了
    elif t == "complete":
        print(f"  客户端收到: type=complete（这是 question.py 第 97 行那种关门信号）")
        if mode == "good":
            return "STOP"
    return None


async def run_session(mode: str):
    log_queue: asyncio.Queue = asyncio.Queue()
    client_stopped = asyncio.Event()

    async def log_pusher():
        while not client_stopped.is_set():
            try:
                entry = await asyncio.wait_for(log_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                if log_queue.empty() and client_stopped.is_set():
                    break
                continue
            # 模拟 websocket.send_json 的网络延迟
            await asyncio.sleep(SEND_DELAY_SEC)
            if client_stopped.is_set() and mode == "bad":
                log_queue.task_done()
                break
            action = await slow_client_receive(entry, mode)
            if action == "STOP":
                client_stopped.set()
            log_queue.task_done()

    pusher = asyncio.create_task(log_pusher())

    async def producer():
        # 按「可能」的入队顺序：两个 result，再 progress(complete)，最后 type=complete
        # （真实业务里，complete 的入队更晚 = 在 generate 返回之后，见 question.py 第 97 行）
        for i in (1, 2):
            await log_queue.put({"type": "result", "id": i})
        await log_queue.put({"type": "progress", "stage": "complete", "completed": 2})
        await asyncio.sleep(0)  # 让 pusher 有机会先消化一部分
        await log_queue.put({"type": "complete", "summary": {"n": 2}})

    await producer()
    await log_queue.join()
    pusher.cancel()
    try:
        await pusher
    except asyncio.CancelledError:
        pass


async def main():
    print("=== 1) 错误示范：一见到 progress.stage=complete 就「关连接」===")
    print("   （后面队列里可能还有 result 没「发到客户端」，本例里你会看到 pusher 仍可能发完，")
    print("   但 bad 模式会提前设 STOP，逻辑上就是丢包/不再处理）\n")
    t0 = time.perf_counter()
    await run_session("bad")
    print(f"耗时约 {time.perf_counter() - t0:.2f}s\n")

    print("=== 2) 正确示范：只认 type=complete 再结束 ===\n")
    t0 = time.perf_counter()
    await run_session("good")
    print(f"耗时约 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())