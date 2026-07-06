"""core/llm/vision_describe.py 单元测试 —— describe_images_into 统一入口。

覆盖两阶段图片描述的 5 条分支（chat / deep_solve / deep_research / quiz 共用）：
  1. 无图 → 原样返回
  2. 主模型原生支持 vision → 跳过（附件保留，交 loop Stage-1 乐观注入）
  3. 描述成功 → 描述拼入文案、图片附件移除
  4. 无可用 vision 模型 → 原样返回、附件不动
  5. 描述全失败 → 原样返回、附件不动（交 loop Stage-2 剥图兜底）
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# 预触发 describe_images_into 内「函数级动态 import」的模块顶层初始化，使其用真实
# settings（conftest env）完成；否则 patch get_settings 期间首次 import 会拿到残缺的
# fake settings（如 langsmith_trace.py 取 langsmith_api_key 失败）。
import core.llm.capabilities  # noqa: F401
import core.observability  # noqa: F401

from core.attachment import from_image_path
from core.llm.vision_describe import describe_images_into


def _make_image(tmp_path: Path) -> str:
    """造一个最小伪 PNG（内容任意，本测试 mock 了真正读图，只需 is_image() 为真）。"""
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    return str(p)


def _ctx(*atts):
    """轻量 fake context：describe_images_into 只读写 attachments 属性。"""
    return SimpleNamespace(attachments=list(atts))


def _fake_settings():
    """主模型 = deepseek（不支持 vision），与 .env 生产配置一致。"""
    return SimpleNamespace(
        llm=SimpleNamespace(text_model="deepseek-chat", binding="deepseek"),
    )


@pytest.fixture
def img(tmp_path):
    return from_image_path(_make_image(tmp_path))


async def test_no_images_returns_base_unchanged(img):
    """无图 → 原样返回 base_text，且不解析 vision 运行时。"""
    ctx = _ctx()  # 无附件
    with patch("settings.get_settings", return_value=_fake_settings()), \
         patch("core.llm.vision_describe.resolve_vision_runtime", new=AsyncMock()) as rv:
        out = await describe_images_into(ctx, "你好")
    assert out == "你好"
    rv.assert_not_called()


async def test_main_model_supports_vision_skips(img):
    """主模型原生支持 vision → 跳过两阶段，图片附件保留。"""
    ctx = _ctx(img)
    with patch("settings.get_settings", return_value=_fake_settings()), \
         patch("core.llm.capabilities.supports_vision", return_value=True), \
         patch("core.llm.vision_describe.resolve_vision_runtime", new=AsyncMock()) as rv:
        out = await describe_images_into(ctx, "看图")
    assert out == "看图"
    assert ctx.attachments == [img]
    rv.assert_not_called()


async def test_describe_appended_and_image_cleared(img):
    """主模型不支持 vision + 有 vision 模型 + 描述成功 → 描述拼入文案、图片附件移除。"""
    ctx = _ctx(img)
    with patch("settings.get_settings", return_value=_fake_settings()), \
         patch("core.llm.capabilities.supports_vision", return_value=False), \
         patch("core.llm.vision_describe.resolve_vision_runtime",
               new=AsyncMock(return_value=(object(), "qwen-vl-plus"))), \
         patch("core.llm.vision_describe.describe_image_attachments",
               new=AsyncMock(return_value=["图里画了一个红色三角形"])):
        out = await describe_images_into(ctx, "看图")
    assert "看图" in out
    assert "红色三角形" in out
    assert ctx.attachments == []  # 图片已移除，loop 不再注入


async def test_no_vision_runtime_keeps_image(img):
    """主模型不支持 vision 但无可用 vision 模型 → 原样返回、附件不动。"""
    ctx = _ctx(img)
    with patch("settings.get_settings", return_value=_fake_settings()), \
         patch("core.llm.capabilities.supports_vision", return_value=False), \
         patch("core.llm.vision_describe.resolve_vision_runtime",
               new=AsyncMock(return_value=(None, None))), \
         patch("core.llm.vision_describe.describe_image_attachments", new=AsyncMock()) as desc:
        out = await describe_images_into(ctx, "看图")
    assert out == "看图"
    assert ctx.attachments == [img]
    desc.assert_not_called()


async def test_all_descriptions_empty_keeps_image(img):
    """vision 模型在但描述全失败（空串占位）→ 原样返回、附件不动（loop Stage-2 剥图兜底）。"""
    ctx = _ctx(img)
    with patch("settings.get_settings", return_value=_fake_settings()), \
         patch("core.llm.capabilities.supports_vision", return_value=False), \
         patch("core.llm.vision_describe.resolve_vision_runtime",
               new=AsyncMock(return_value=(object(), "qwen-vl-plus"))), \
         patch("core.llm.vision_describe.describe_image_attachments",
               new=AsyncMock(return_value=[""])):
        out = await describe_images_into(ctx, "看图")
    assert out == "看图"
    assert ctx.attachments == [img]
