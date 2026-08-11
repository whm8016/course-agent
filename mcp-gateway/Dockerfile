# MCP gateway sidecar：把 stdio 类 MCP server 收敛成一份 SSE 服务，供 backend 多 worker 共享。
#
# 为什么存在：gunicorn -w 4 下，每个 worker 各起一份 uvx stdio 子进程 = 4N 份常驻进程，
# 白吃内存。本容器只跑 1 份 mcp-proxy + 每个 server 各 1 份子进程，backend 的 4 个 worker
# 全改成 HTTP 连这一个 gateway（4→1，省内存），且 stdio 子进程全网只剩这一份。
#
# 方案 A（构建期预装死版本，运行时零网络）：uv tool install 在 build 期把各 server 钉死
# 版本装进独立 venv，运行时直接 exec 已装好的入口命令，不再 uvx 运行时拉包。版本变更必须
# 走一次 build → 可复现、可回滚、可审计（对冲运行时 @latest 的 installer spoofing 风险）。
FROM python:3.12-slim-bookworm

# uv 二进制（与 backend/Dockerfile 同款：从官方镜像复制，免单独安装）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv tool install 把每个包装进独立 venv，入口落到 UV_TOOL_BIN_DIR；运行时 PATH 直接命中。
# UV_PYTHON_DOWNLOADS=never：只用 base image 自带的 python3.12，build 期不下其它解释器。
ENV UV_TOOL_BIN_DIR=/opt/uv-bin \
    UV_PYTHON_DOWNLOADS=never

# 国内 build 经 --build-arg UV_INDEX_URL 指镜像源（如清华）加速；不传用 PyPI 默认。
# 三个包各自独立 tool venv，互不影响 mcp 版本：
#   - modelscope-mcp-server 依赖 mcp 2.x（fastmcp）
#   - mcp-server-fetch 需 mcp<2（旧 SDK），用 --with 注入约束
#   - mcp-proxy 自身独立
ARG UV_INDEX_URL
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_URL="${UV_INDEX_URL:-}" uv tool install modelscope-mcp-server==0.2.1 \
    && UV_INDEX_URL="${UV_INDEX_URL:-}" uv tool install --with "mcp<2" mcp-server-fetch \
    && UV_INDEX_URL="${UV_INDEX_URL:-}" uv tool install --with "mcp<2" mcp-proxy==0.12.0

ENV PATH="/opt/uv-bin:${PATH}"

EXPOSE 8096
# mcp-proxy 把多个 stdio server 暴露成命名 SSE 端点：/servers/{name}/sse。
# --pass-environment：把容器 env（含 MODELSCOPE_API_TOKEN、PATH）透传给 stdio 子进程。
# backend 侧每个 server 配 type=sse + url=http://mcp-gateway:8096/servers/{name}/sse。
CMD ["mcp-proxy", \
     "--host", "0.0.0.0", \
     "--port", "8096", \
     "--pass-environment", \
     "--named-server", "modelscope", "modelscope-mcp-server", \
     "--named-server", "fetch", "mcp-server-fetch"]
