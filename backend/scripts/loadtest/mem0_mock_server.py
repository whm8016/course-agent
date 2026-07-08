"""mem0 外部依赖 mock server：零依赖（纯标准库 http.server），喂给 mem0 的 embedding + LLM。

为什么需要它：
  mem0 记忆功能内部要调两个外部 API——embedding（文字→向量，存 pgvector）和 LLM
  （记忆事实提取）。压测环境没有可用的 embedding key（DeepSeek 无 embedding 接口），
  占位 key 会让 mem0 401 重试 ~5s 阻塞 first token。本 server 用确定性假响应接管这两个调用。

仿真边界（不是关掉 mem0）：
  mem0 自身的 pgvector 存取 / 记忆 CRUD / flush / 并发——全部真实跑（被测对象）。
  只有 mem0 调的两个外部 API 被换成 mock（外部依赖，符合压测惯例）。

确定性向量：同一 query 永远返回同一 1024 维向量（sha256 派生 + 归一化）。
不同 query 向量不同，pgvector 的余弦检索语义仍可区分（虽非真实语义，但压测不关心检索
质量，只关心 mem0 链路的延迟/并发/资源）。维度 1024 严格匹配 pgvector collection dim
（settings.lightrag.embedding_dim），否则 mem0 写库报维度不匹配。

容器内 8080，compose service 名 mem0-mock，mem0 经 http://mem0-mock:8080/v1 访问。
"""
from __future__ import annotations

import hashlib
import json
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer

EMB_DIM = 1024  # 必须等于 settings.lightrag.embedding_dim（pgvector collection dim）


def _det_vector(text: str) -> list[float]:
    """基于文本 sha256 的确定性 1024 维向量（归一化），不同文本不同向量。"""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((EMB_DIM // len(h)) + 1))[: EMB_DIM * 4]
    vals = list(struct.unpack(f"<{EMB_DIM}f", raw))
    norm = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / norm for v in vals]


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        path = self.path
        if "embeddings" in path:
            inputs = body.get("input", "")
            # mem0/openai 可能传 str / list[str] / list[list[float]]（已向量化）
            if isinstance(inputs, str):
                inputs = [inputs]
            elif isinstance(inputs, list) and inputs and isinstance(inputs[0], list):
                inputs = [str(x) for x in inputs]
            self._send(200, {
                "object": "list",
                "data": [{"object": "embedding", "index": i,
                          "embedding": _det_vector(str(t))}
                         for i, t in enumerate(inputs)],
                "model": body.get("model", "text-embedding-mock"),
                "usage": {"prompt_tokens": max(1, len(inputs)), "total_tokens": max(1, len(inputs))},
            })
        elif "chat/completions" in path:
            # mem0 add 用 LLM 提取记忆事实：返回中性内容，mem0 多半判 NOOP（不新增），
            # 避免 mock 持续往 pgvector 灌假记忆撑大库。
            self._send(200, {
                "id": "mock-chat",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": "无新增事实"},
                             "finish_reason": "stop"}],
                "model": body.get("model", "mock-llm"),
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            })
        else:
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

    def do_GET(self):  # noqa: N802
        if self.path == "/" or "health" in self.path:
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):  # 静默，不打日志（避免淹没 backend 日志）
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
