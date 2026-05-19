# Deep Research 渐进开发记录

## Step 0 — 2026-05-19

- 改了什么：归档 `_legacy/`，主路径 single-shot：`topic -> RAG -> LLM 报告`
- test 命令：`cd backend` → `python -m uvicorn main:app --reload --port 8000` → `python test_ws_deep_research.py`
- 观察到的问题：本机 DashScope 账户欠费导致真实 LLM 调用 400；用 mock 验证 pipeline 六段 progress + md 落盘正常。uvicorn 需完整 backend 依赖（slowapi 等）。
- 下一步改进：Step 1 多轮 Research 循环
