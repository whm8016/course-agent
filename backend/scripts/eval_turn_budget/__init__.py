"""走完整 turn 的上下文预算 A/B 评测包。

对比「线上默认 context_policy」与「coordinator + priority」两套上下文预算方案，**必须经
``turn_runtime.start_turn``** 走完整 turn 链路（不能用直调 orchestrator 的 solver，也不能复用
eval_context 的 ``set_arm`` runner--``set_arm`` 会短路 coordinator 分支）。

- ``config.py``     目录常量 + ARMS 两臂 settings 覆写定义
- ``runner.py``     单 case 单臂执行器（patch settings -> start_turn -> subscribe -> 落盘）
- ``run_eval.py``   CLI 入口 + 汇总对比
- ``datasets/``     多轮评测集（history 跨过 resolve_budget，否则裁剪不触发、评测无意义）
"""
