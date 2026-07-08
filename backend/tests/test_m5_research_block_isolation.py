"""M-5：research 并行 block context 浅拷贝污染 → 深拷贝隔离。

修复前：dataclasses.replace 浅拷贝，attachments / metadata 仍指向原 context 同一对象。
research 的 _research_block 多块并行（asyncio.gather），run_agent_loop→_build_messages 会
原地改 Attachment（doc 文件清空 base64 释放内存）。共享同一份时，先跑的 block 清空 base64，
后跑的 block 看到的 base64 已是 None，附件内容丢失。

修复后：_fork_for_block 对 attachments / metadata 深拷贝，每块拿到独立副本，互不污染。
"""
from __future__ import annotations

import asyncio

from core.attachment import Attachment, AttachmentType
from core.context import UnifiedContext
from core.research.pipeline import _fork_for_block


def _ctx_with_doc_attachment() -> UnifiedContext:
    """构造带 1 个 doc 文件附件（带 base64）的 context。"""
    att = Attachment(
        type=AttachmentType.FILE,
        filename="note.pdf",
        base64="SGVsbG8gV29ybGQ=",  # "Hello World"
    )
    return UnifiedContext(
        course_id="C1",
        user_id="U1",
        user_message="研究主题",
        attachments=[att],
        metadata={"shared_key": "original"},
    )


def test_fork_deep_copies_attachments():
    """fork 后修改 block 的附件 base64 不影响原 context 的附件。"""
    ctx = _ctx_with_doc_attachment()
    forked = _fork_for_block(ctx, user_message="block task", conversation_history=[])

    # 模拟 _build_messages 的原地修改：清空 base64
    forked.attachments[0].base64 = None

    # forked 的附件 base64 被清空
    assert forked.attachments[0].base64 is None
    # 原始 context 的附件 base64 不受影响（深拷贝隔离）
    assert ctx.attachments[0].base64 == "SGVsbG8gV29ybGQ="
    # 不是同一个对象（深拷贝产生了独立 Attachment）
    assert forked.attachments[0] is not ctx.attachments[0]


def test_fork_deep_copies_metadata():
    """fork 后写 block 的 metadata 不影响原 context / 其它 block 的 metadata。"""
    ctx = _ctx_with_doc_attachment()
    a = _fork_for_block(ctx, mode="research")
    b = _fork_for_block(ctx, mode="research")

    a.metadata["block_id"] = "A"
    b.metadata["block_id"] = "B"

    assert a.metadata["block_id"] == "A"
    assert b.metadata["block_id"] == "B"
    # 原始 context 的 metadata 未被污染（浅拷贝会三者指向同一 dict，全是 "B"）
    assert "block_id" not in ctx.metadata
    assert ctx.metadata["shared_key"] == "original"


def test_parallel_blocks_do_not_cross_contaminate_attachments():
    """模拟两个 block 并行：各自清空自己副本的 base64，互不影响，原 context 完好。

    推演（M-5 interleaving）：
      - 正常：两个 block 各自 deepcopy，独立改自己的副本 → 原 ctx + 两副本互不影响。
      - 竞态（修复前）：两 block 共享同一 Attachment，A 先清空 → B 看到空 base64（污染）。
      - 异常：某 block 抛异常，其副本被 GC，不影响另一 block（深拷贝无共享引用）。
    """
    ctx = _ctx_with_doc_attachment()
    forked_list = [_fork_for_block(ctx, mode="research") for _ in range(3)]

    async def _simulate_clear_base(forked_ctx):
        # 模拟 _build_messages 清空 doc 附件 base64（耗时操作，可能交错）
        await asyncio.sleep(0)
        for a in forked_ctx.attachments:
            if not a.is_image() and a.base64:
                a.base64 = None
        return forked_ctx.attachments[0].base64

    async def _go():
        return await asyncio.gather(*[_simulate_clear_base(f) for f in forked_list])

    results = asyncio.run(_go())

    # 三个 block 各自清空了自己副本的 base64 → 各自 None
    assert all(r is None for r in results)
    # 原 context 的附件 base64 完好（核心断言：无共享污染）
    assert ctx.attachments[0].base64 == "SGVsbG8gV29ybGQ="
    # 三个 fork 副本互不相同
    assert forked_list[0].attachments[0] is not forked_list[1].attachments[0]
    assert forked_list[1].attachments[0] is not forked_list[2].attachments[0]


def test_fork_preserves_other_fields_via_replace():
    """fork 仍走 replace 语义：其余字段（course_id/user_id 等）正常透传 + override 生效。"""
    ctx = _ctx_with_doc_attachment()
    forked = _fork_for_block(
        ctx, user_message="新任务", conversation_history=[], enabled_tools=["rag"], mode="research"
    )
    assert forked.course_id == "C1"
    assert forked.user_id == "U1"
    assert forked.user_message == "新任务"
    assert forked.conversation_history == []
    assert forked.enabled_tools == ["rag"]
    assert forked.mode == "research"
