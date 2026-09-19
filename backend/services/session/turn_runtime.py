"""
Turn Runtime Manager
====================

管理单回合（turn）的完整生命周期：创建、后台执行、事件订阅、取消。

【在架构中的位置】
  API 层（POST /api/chat、WS /api/run/{cap}）
    → TurnRuntimeManager.start_turn(context)         返回 turn_id
    → TurnRuntimeManager.subscribe_turn(turn_id)     订阅事件流（支持 after_seq 重连）
    → TurnRuntimeManager.cancel_turn(turn_id)        取消（客户端断开时）

  内部执行链：
    _run_turn()
      → ContextBuilder.build(history)                裁剪历史到 token 预算内
      → async for event in orchestrator.handle(ctx): 消费 orchestrator 事件流
          await execution.bus.emit(event)            fan-out 给所有 subscribe_turn 订阅者
      → 收集 ANSWER 内容 → 发布 CAPABILITY_COMPLETE 到全局 EventBus

【TurnRuntimeManager 对应关系】
  start_turn()      ← start_turn()
  subscribe_turn()  ← subscribe_turn(turn_id, after_seq)（in-memory 回放）
  cancel_turn()     ← cancel_turn()
  _TurnExecution    ← _TurnExecution（turn_id / bus / task / seq 事件列表）
  EventBus 发布     ← _publish_live_event + CAPABILITY_COMPLETE

  本版本暂不含：
  - SQLite 事件持久化（after_seq 基于内存事件列表实现）
  - regenerate_last_turn
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from core.context import UnifiedContext
from core.observability import bind_context, log_flow
from core.observability.metrics import observe_turn, observe_turn_error
from core.stream import StreamEvent, StreamEventType
from core.stream_bus import StreamBus
from services.session import reply_channel
from services.session.context_builder import ContextBuilder, resolve_budget
from settings import get_settings
TEXT_MODEL = get_settings().llm.text_model

logger = logging.getLogger(__name__)

_EXECUTION_TTL = 300.0  # turn 结束后保留状态的时间（秒），供迟到订阅者回放
_ANSWER_NOW_POLL_S = 1.0  # 跨 worker answer_now watcher 轮询间隔（信号最坏投递延迟）


@dataclass
class _TurnExecution:
    """单个 turn 的运行时状态。"""

    turn_id: str
    context: UnifiedContext
    bus: StreamBus = field(default_factory=StreamBus)
    task: asyncio.Task | None = None
    # (seq, event) 列表，供 subscribe_turn(after_seq) 做内存回放
    events: list[tuple[int, StreamEvent]] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    # answer_now（立即回答）信号不挂在这里：它和 ask_user 的回复通道（reply_channel）一样
    # 由 manager 在 start_turn 创建 execution 之前就新建并注入 context.metadata，因此无法
    # 在 _TurnExecution 构造时传入。


# 取消轮落盘的尾注，与前端 ChatWindow 停止生成时的展示格式逐字一致（重载回放不跳变）。
_STOP_NOTE = "_（已停止生成）_"
_STOP_EMPTY_NOTE = "_（已停止生成，未产生回答）_"


def _is_persistent_session(session_id: str | None, turn_id: str) -> bool:
    """是否挂在真实持久会话上：start_turn 对无会话 turn 用 turn_id 占位
    session_id（无 Session 行），落盘/溢出兜底等都以本谓词单点判定。"""
    return bool(session_id) and session_id != turn_id


async def _sync_compress_and_reload(turn_id: str, context: UnifiedContext) -> str:
    """溢出兜底一步：当轮同步强制压缩推进 L2 游标，并重载摘要卡文本。

    maybe_compress(force=True) 绕过 token 闸但仍受三层并发锁与 OCC 保护——别的
    worker 正在压同 session 时让路返回 ""（其结果同样会推进游标，本轮退回原裁剪
    方案）。单 DB 会话内压缩 + 重载：读的正是刚 commit 的同一行，省一次会话开闭
    与跨会话重读（该路径内联在首 token 之前，直接计入首字延迟）。
    """
    session_id = context.session_id or ""
    if not _is_persistent_session(session_id, turn_id):
        return ""
    try:
        from core.db.database import AsyncSessionLocal
        from core.memory.session_summary import get_summary_manager

        mgr = get_summary_manager()
        async with AsyncSessionLocal() as db:
            if not await mgr.maybe_compress(db, session_id, force=True):
                return ""
            return await mgr.get_summary(
                db, session_id, user_id=context.user_id or None
            )
    except Exception:
        logger.warning(
            "TurnRuntime: force L2 compress/reload failed turn_id=%s session=%s",
            turn_id, session_id, exc_info=True,
        )
        return ""


def _persist_attachments_meta(context: UnifiedContext) -> list[dict] | None:
    """context.attachments -> user 行 metadata.attachments（前端 AttachmentInfo 形状）。

    与 api/sessions 持久化同口径：只存元数据，base64/file_path（服务端内部路径）不落。
    """
    return [
        a.model_dump(include={"type", "url", "mime_type", "filename"}, exclude_none=True, mode="json")
        for a in (context.attachments or [])
    ] or None


async def _persist_turn_rows(
    execution: _TurnExecution,
    *,
    agent_output: str,
    token_text: str,
    cancelled: bool,
    error_text: str | None,
    error_code: str = "",
    error_ref: str = "",
    done_metadata: dict,
    result_metadata: dict,
    result_questions: list | None,
) -> None:
    """回合结束把 user/assistant 两行写进 messages（服务端权威回合日志）。

    对标 Letta / Agents SDK：历史由服务端持有，不依赖客户端落库。此前前端流结束后
    best-effort saveMessage——关标签页/出错/取消轮永久丢失，且早于客户端落库触发的
    压缩器永远看不到最新一轮。

    幂等：(session_id, turn_id, role) 唯一（add_message 先查短路 + DB 唯一索引兜底）。
    跳过条件：无 session_id 或 session_id==turn_id（start_turn 对无会话 turn 的占位，
    无 Session 行可挂）、Session 行不存在（防 FK 违约）、**归属不符**（session 不属于
    本 turn 的 user_id——session_id 来自请求体，写侧必须与读侧（load_uncompressed_tail
    的 user_id 过滤）同口径防 B 用户向 A 的会话写行）。best-effort：任何异常只告警。

    时序：两行 created_at 严格递增（assistant >= user + 1e-6）——keyset 游标与
    展示排序都按 (created_at, id)，Windows time.time() 背靠背调用相等率高，
    同 ts + 随机 id 会让 assistant 排到 user 前甚至被 keyset 下界永久排除。

    metadata 对齐前端此前写入集合（ChatWindow saveMessage）：mode/tools_used/truncated/
    usage/stopped/quiz + user 行 attachments。intent/intent_confidence/retrieve_mode/
    retrieve_strategy/hallucination/chunks 不落——现行 loop 的 done/tool_result 事件
    本就不产这些字段（前端此前落库也恒空），放弃而非补造。
    """
    ctx = execution.context
    session_id = ctx.session_id or ""
    if not _is_persistent_session(session_id, execution.turn_id):
        return  # 无持久会话（游客/占位 session），无行可落
    try:
        from sqlalchemy import select as _select

        from core.db.database import AsyncSessionLocal, Session as DbSession
        from core.memory.memory import add_message

        async with AsyncSessionLocal() as db:
            # 只取归属 + course_id 两列（与读侧 load_uncompressed_tail 的列裁剪对称，
            # 不白拉 summary Text blob——本路径每回合必经）
            session_row = (await db.execute(
                _select(DbSession.user_id, DbSession.course_id)
                .where(DbSession.id == session_id)
            )).first()
            if session_row is None:
                return  # Session 行不存在（异常路径），跳过避免 FK 违约
            if str(session_row.user_id or "") != str(ctx.user_id or ""):
                logger.warning(
                    "TurnRuntime: persist skipped, session %s not owned by turn user %s",
                    session_id, ctx.user_id,
                )
                return  # 归属不符：写侧与读侧同口径（防越权写他人会话）

            atts = _persist_attachments_meta(ctx)
            ts_user = time.time()
            await add_message(
                db, session_id, "user", ctx.user_message or "", "text",
                metadata={"attachments": atts} if atts else None,
                turn_id=execution.turn_id, created_at=ts_user,
                course_id=session_row.course_id or "",
                touch_session=False,  # 一回合 touch 一次，由 assistant 行收尾
            )

            # assistant 行正文：quiz 能力题目走 metadata.quiz（前端 QuizCard 渲染口径，
            # content 恒空）；取消轮拼 token 部分输出；错误轮记错误文本；其余用完整输出。
            if result_questions is not None:
                content = ""
            elif cancelled:
                partial = agent_output or token_text
                content = f"{partial}\n\n{_STOP_NOTE}" if partial else _STOP_EMPTY_NOTE
            elif error_text is not None:
                content = f"出错了: {error_text}"
            else:
                content = agent_output

            asst_meta: dict = {"mode": done_metadata.get("mode") or ctx.mode or "chat"}
            if done_metadata.get("tools_used"):
                asst_meta["tools_used"] = done_metadata["tools_used"]
            if done_metadata.get("truncated"):
                asst_meta["truncated"] = True
            usage = done_metadata.get("usage") or (ctx.metadata or {}).get("llm_usage")
            if usage:
                asst_meta["usage"] = usage
            if result_metadata:
                asst_meta.update(result_metadata)
            if result_questions is not None:
                asst_meta["quiz"] = {"questions": result_questions}
            if cancelled:
                asst_meta["stopped"] = True
            if error_text is not None:
                # 错误信封元数据：重载会话时前端按此恢复错误样式与报障编号
                #（content 保持「出错了: 」前缀，兼容旧读取方）
                asst_meta["error"] = {"code": error_code or "internal", "ref": error_ref}
            # assistant 行时间戳严格大于 user 行（同回合顺序由时间戳而非随机 id 决定）
            await add_message(
                db, session_id, "assistant", content, "text",
                metadata=asst_meta, turn_id=execution.turn_id,
                created_at=max(time.time(), ts_user + 1e-6),
                course_id=session_row.course_id or "",
            )
            await db.commit()
        logger.info(
            "TurnRuntime: persisted turn rows turn_id=%s session=%s cancelled=%s error=%s",
            execution.turn_id, session_id, cancelled, bool(error_text),
        )
    except Exception:
        logger.warning(
            "TurnRuntime: persist turn rows failed turn_id=%s session=%s",
            execution.turn_id, session_id, exc_info=True,
        )


async def _persist_turn_error(
    execution: _TurnExecution,
    *,
    error_code: str,
    error_ref: str,
    message: str,
    exc_type: str,
    exc_head: str,
) -> None:
    """失败 turn 落 turn_errors 流水行（append-only 运维读模型，best-effort 不阻塞 turn）。

    与 messages 落盘解耦：错误流水是 admin 排障数据（GET /admin/errors/recent 按学生
    报障 ref 精确查），失败也不影响叙事流落盘。无 FK 裸字符串——用户/课程删除后行须
    存活（对齐 llm_usage_records）。
    """
    ctx = execution.context
    try:
        from core.db.database import AsyncSessionLocal, TurnErrorRecord

        async with AsyncSessionLocal() as db:
            db.add(TurnErrorRecord(
                turn_id=execution.turn_id,
                ref=(error_ref or execution.turn_id[:8])[:16],
                session_id=ctx.session_id or "",
                user_id=str(ctx.user_id or ""),
                course_id=str(ctx.course_id or ""),
                mode=ctx.mode or "chat",
                code=error_code or "internal",
                message=message[:512],
                exc_type=exc_type[:128],
                exc_head=exc_head[:256],
            ))
            await db.commit()
    except Exception:
        logger.warning(
            "TurnRuntime: persist turn error row failed turn_id=%s",
            execution.turn_id, exc_info=True,
        )


class TurnRuntimeManager:
    """管理 turn 生命周期：创建 → ContextBuilder → orchestrator → fan-out → EventBus。"""

    def __init__(self) -> None:
        self._executions: dict[str, _TurnExecution] = {}
        self._lock = asyncio.Lock()
        # ask_user 暂停/恢复经 reply_channel（Redis BLPOP/RPUSH 跨 worker；memory:// 回退进程内），
        # 状态不再挂在 manager 上——见 services.session.reply_channel。
        # answer_now 信号：turn_id → asyncio.Event，前端"立即回答"按钮 set()，loop 每轮顶部
        # 轮询（与 reply_channel 同生命周期：start_turn 建，_run_turn finally 清）。
        # 跨 worker 投递见 reply_channel.set_answer_now 与 _watch_answer_now_flag。
        self._answer_now_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    async def start_turn(self, context: UnifiedContext) -> str:
        """启动一个新 turn，返回 turn_id。

        后台任务：调用 ContextBuilder 裁剪历史 → 驱动 orchestrator 生成事件流。
        """
        turn_id = uuid.uuid4().hex  # 32 位无横杠——需塞进 memory_episodes.turn_id VARCHAR(32)
        if not context.session_id:
            context.session_id = turn_id
        context.metadata["turn_id"] = turn_id  # orchestrator 用于注册全局 bus

        # 历史裁剪：coordinator_enabled 时走 plan_turn（MECW 感知 + 可选摘要续接，Phase 3），
        # 否则旧 ContextBuilder + resolve_budget（20% 窗口，逐字节同旧行为）。
        # 埋点：两条分支都记 before/after 计数——之前这里是全哑的，LangSmith 和应用日志都看不到
        # 任何裁剪痕迹，只能靠事后临时脚本翻 ctx.metadata 才能确认真的裁了。
        # 预算按**本轮 profile 模型**算（旧 bug：恒用全局 TEXT_MODEL——用户选长窗口模型会裁
        # 过狠、选小模型裁不够）。resolve_profile_runtime 读 DB/Redis 缓存，失败回退 TEXT_MODEL。
        _turn_model = TEXT_MODEL
        try:
            from core.pipeline_common import resolve_profile_runtime

            _rt = await resolve_profile_runtime(context.llm_profile_id, context.user_id)
            _turn_model = _rt.text_model or TEXT_MODEL
        except Exception:
            logger.debug("resolve_profile_runtime failed, fallback TEXT_MODEL", exc_info=True)
        _history_before = len(context.conversation_history)
        _history_snapshot = list(context.conversation_history)  # 被砍正文差集的裁剪前快照
        if get_settings().context_budget.coordinator_enabled:
            from core.agentic.context_budget import plan_turn
            _plan = plan_turn(
                history=context.conversation_history,
                model=_turn_model,
                session_summary_text=context.session_summary or "",
            )
            # 溢出当轮同步兜底（对齐 Letta 溢出时先 summarize_messages_inplace 再重试）：
            # 读侧只装游标之后的消息，裁掉即越游标——被裁消息既不在摘要卡里也不在本轮
            # 原文里，违反不变式。此时同步强制压缩推进游标（被裁消息进卡），再按新卡
            # 重新裁剪。正常轮不触发（未裁不动），零额外延迟。一次性不循环：窗口内的
            # 近消息压缩也救不了（boundary 保留窗口），残余溢出交给轮内 enforce 级联。
            if _plan.dropped_beyond_cursor and _is_persistent_session(context.session_id, turn_id):
                _summary2 = await _sync_compress_and_reload(turn_id, context)
                if _summary2:
                    _plan = plan_turn(
                        history=context.conversation_history,  # 原始 history（尚未被裁剪结果覆盖）
                        model=_turn_model,
                        session_summary_text=_summary2,
                    )
                    # 摘要切片换重载后的新卡：下方 carry 前插与 preview 捕获都读
                    # context.session_summary，不换会用到旧卡（埋点陈旧一轮）
                    context.session_summary = _summary2
                else:
                    # 成因不逐一区分（maybe_compress 返回 False 的各种路径：锁让路/
                    # 条数守卫拒压/压缩或重载失败/重载为空）。按原裁剪方案走，被裁
                    # 消息当轮不可见——记 WARN 可观测（下轮读侧触发会再压）
                    logger.warning(
                        "TurnRuntime: overflow fallback ineffective (compress "
                        "declined/failed or reload empty); turn_id=%s trimmed "
                        "messages beyond cursor uncovered this round",
                        turn_id,
                    )
            context.conversation_history = _plan.keep_history
            context.metadata["_budget_plan"] = _plan
            if _plan.carry_forward_added:
                # 前插的摘要正文 preview 先捕获再清空（供 pretrim 子 run 上报）
                context.metadata["_ctx_pretrim_summary_preview"] = (context.session_summary or "")[:800]
                context.session_summary = ""  # 摘要已前插历史，清 system 切片避免双付
            context.metadata["_ctx_pretrim"] = {
                "coordinator_enabled": True,
                "history_before": _history_before,
                "history_after": len(context.conversation_history),
                "dropped": _plan.dropped_count,
                "carry_forward_added": _plan.carry_forward_added,
                "mecw": _plan.mecw,
                "target_input_tokens": _plan.target_input_tokens,
            }
        else:
            builder = ContextBuilder(max_history_tokens=resolve_budget(_turn_model))
            context.conversation_history = builder.build(context.conversation_history)
            context.metadata["_ctx_pretrim"] = {
                "coordinator_enabled": False,
                "history_before": _history_before,
                "history_after": len(context.conversation_history),
                "dropped": _history_before - len(context.conversation_history),
                "carry_forward_added": False,
            }
        # 被砍历史正文 preview（对象 identity 差集；ContextBuilder.build / plan_turn 的
        # keep_history 都保留原消息对象引用）。供 _run_turn 里 context.trim.pretrim 子 run
        # 上报——此前回合前裁剪只有计数挂在 turn 根 run metadata，删了什么内容无从查证。
        # 零裁剪轮跳过（差集白算，且 _run_turn 只在 dropped>0 时读此键）；preview 格式
        # 与 ③收缩循环丢轮共用 context_budget.msg_preview，钉住同一形态防漂移。
        if context.metadata["_ctx_pretrim"]["dropped"]:
            from core.agentic.context_budget import msg_preview

            _kept_ids = {id(m) for m in context.conversation_history}
            context.metadata["_ctx_pretrim_dropped_previews"] = [
                msg_preview(m) for m in _history_snapshot if id(m) not in _kept_ids
            ][:10]
        log_flow("turn.context_pretrim", turn_id=turn_id, **context.metadata["_ctx_pretrim"])

        # ask_user 暂停/恢复经 reply_channel（Redis BLPOP/RPUSH，跨 worker 投递；memory:// 回退进程内）。
        # loop 调用 ask_user 工具时 await context.metadata["wait_for_user_reply"]().
        async def _wait_for_user_reply() -> dict | None:
            # 硬超时：clarify_enabled 时取 settings.research.clarify_wait_timeout_s，超时返回 skip
            # payload（loop._format_reply → "User skipped."，loop 续跑），绝不返回 None（None 会被
            # loop 当作 turn 取消而 break）。clarify_enabled=False → timeout=0 无限等（旧「无限等」）。
            _cfg = get_settings().research
            timeout = _cfg.clarify_wait_timeout_s if _cfg.clarify_enabled else 0
            raw = await reply_channel.wait_reply(turn_id, timeout)
            if raw is None:
                logger.info(
                    "TurnRuntime: ask_user 等待超时/无回复 turn_id=%s（timeout=%ss），走 skip 续跑",
                    turn_id, timeout,
                )
                return {"text": "", "answers": None}
            return raw

        context.metadata["wait_for_user_reply"] = _wait_for_user_reply
        # turn 归属 key：跨 worker submit_user_reply 本地 _executions miss 时回落它做 IDOR 校验
        await reply_channel.set_turn_owner(turn_id, str(context.user_id or ""))

        # answer_now 信号：注入 is_answer_now_requested 回调（返回 bool，非协程），
        # run_agent_loop 每轮顶部轮询它决定是否提前直接回答。仿 wait_for_user_reply 注入模式。
        answer_now_ev = asyncio.Event()
        self._answer_now_events[turn_id] = answer_now_ev
        context.metadata["is_answer_now_requested"] = answer_now_ev.is_set

        # bind_context BEFORE create_task：asyncio 会把当前 context 复制给子任务
        bind_context(
            turn_id=turn_id,
            user_id=str(context.user_id or ""),
            course_id=str(context.course_id or ""),
            mode=str(context.mode or ""),
        )
        log_flow("turn.start", turn_id=turn_id, mode=context.mode,
                 user_id=context.user_id, course_id=context.course_id)

        # 把 turn_id 写进 context.metadata，供 loop 层的 record_llm_usage 回溯落库明细（loop 只能
        # 读 context，看不到 turn_runtime 的 execution）。turn_id 缺失不影响聚合（明细按 day rollup）。
        context.metadata["turn_id"] = turn_id

        execution = _TurnExecution(turn_id=turn_id, context=context)

        async with self._lock:
            self._executions[turn_id] = execution

        execution.task = asyncio.create_task(
            self._run_turn(execution),
            name=f"turn-{turn_id[:8]}",
        )
        return turn_id

    async def subscribe_turn(
        self,
        turn_id: str,
        after_seq: int = 0,
        *,
        user_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """订阅指定 turn 的事件流，支持断线后按 seq 回放。

        Args:
            turn_id:   由 start_turn() 返回的 ID。
            after_seq: 从第几个事件开始（0 = 全部）。已完成的事件从内存列表回放，
                       新事件从 fan-out bus 实时接收。
            user_id:   可选，发起订阅的当前登录用户 id。提供时做 turn 归属校验：
                       turn 不属于该用户则视为不存在（KeyError），防止 B 重连订阅
                       A 的 turn_id 偷看事件流（IDOR）。

        Raises:
            KeyError: turn_id 不存在，或归属校验未通过。
        """
        execution = self._executions.get(turn_id)
        if execution is None or not self._turn_belongs_to(execution, user_id):
            raise KeyError(f"未知 turn_id: {turn_id}")

        seq = 0
        async for event in execution.bus.subscribe():
            if seq >= after_seq:
                yield event
            seq += 1

    async def cancel_turn(self, turn_id: str) -> None:
        """取消正在运行的 turn（客户端断开时调用）。"""
        execution = self._executions.get(turn_id)
        if execution is None:
            return
        if execution.task and not execution.task.done():
            execution.task.cancel()
            log_flow("turn.cancel", turn_id=turn_id)

    async def submit_user_reply(
        self,
        turn_id: str,
        text: str | None = None,
        answers: list[dict] | None = None,
        outline: list[dict] | None = None,
        *,
        user_id: str | None = None,
    ) -> bool:
        """向正在等待 ask_user 回复的 turn 投递用户回答。

        投递走 reply_channel（Redis RPUSH，跨 worker；memory:// 回退进程内队列）。

        Args:
            text/answers: ask_user 工具的常规回复（文本 / 选项回答）。
            outline: 深度研究大纲确认的编辑结果（list[{title, overview}]）；与 ask_user 共用同一
                     回复通道，pipeline 侧用 reply.get("outline") 取。三者互斥使用，按触发场景给。
            user_id: 可选，发起回复的当前登录用户 id。提供时做 turn 归属校验：本地 _executions
                     命中走 _turn_belongs_to；跨 worker（本地 miss）回落 Redis owner key
                     （ca:turn:owner:{turn_id}）比对，防止 B 用户拿 A 的 turn_id 向 A 的 ask_user
                     投递回复、操纵 A 的对话（Turn IDOR，与 answer_now 同源）。

        Returns:
            True  — 归属校验通过并已投递（best-effort：对方 worker 是否在等不保证，无消费者则 TTL 自清理）
            False — 归属校验未通过（turn 不存在/已结束，或不属于该 user_id）
        """
        if user_id is not None:
            execution = self._executions.get(turn_id)
            if execution is not None:
                # 本进程持有该 turn：直接比对 context.user_id
                if not self._turn_belongs_to(execution, user_id):
                    logger.warning(
                        "TurnRuntime: submit_user_reply turn_id=%s 归属不符", turn_id
                    )
                    return False
            else:
                # 跨 worker：本地无 execution，回落 Redis owner key 做 IDOR 校验
                if not await self._cross_worker_owner_ok(turn_id, user_id, "submit_user_reply"):
                    return False
        payload: dict = {"text": text or "", "answers": answers, "outline": outline}
        await reply_channel.push_reply(turn_id, payload)
        logger.info("TurnRuntime: submitted user reply for turn_id=%s", turn_id)
        return True

    async def request_answer_now(
        self, turn_id: str, *, user_id: str | None = None
    ) -> bool:
        """触发"立即回答"：让正在运行的 turn 的 agent loop 在下一轮顶部提前直接回答。

        信号投递两级（仿 submit_user_reply 的双级归属校验 + 回落；SSE 长连接与按钮
        POST 在 gunicorn -w4 下常落不同 worker）：
        1. 本进程持有 turn（_answer_now_events 命中）→ 直接 ev.set()（幂等，连点安全），
           不碰 Redis——同 worker 快路径零新增延迟。
        2. 本地 miss（turn 在其他 worker）→ Redis owner key（ca:turn:owner:{turn_id}）
           做 IDOR 校验后 SET 标志位 ca:answer_now:{turn_id}；持有 turn 的 worker 上
           _watch_answer_now_flag 轮询到标志 → 置同一把本地 ev（loop.py 零改动，
           最坏延迟 ≈ _ANSWER_NOW_POLL_S）。无 Redis（memory://）跨 worker 投递不达，
           返回 False（= 单 worker 语义，与修复前一致）。

        边界：execution 在而 ev 不在（turn 已结束、execution 还在 _EXECUTION_TTL=300s
        回放窗内）必须返回 False 且**严禁落入 Redis 路径**——owner key TTL 600s 未过期，
        落进去会放行并谎报 200，让跨 worker 行为比修复前（同 worker 此时 False→404）更差，
        还给死 turn 留 600s 垃圾 flag。此分支与 user_id 无关。
        残窗（best-effort，与 push_reply 对已结束 turn 投递同语义）：turn 已结束但 owner key
        （TTL 600s 自 turn 开始计）未过期、且请求落在非持有 worker（或 execution 已过 300s
        被逐出）时，跨 worker 路径仍放行返回 True——投递成功 ≠ turn 仍存活。前端流结束即
        隐藏按钮，实际不可达；彻底关闭须 turn 结束时 DEL owner key，牵动 submit_user_reply
        契约与 run.py 订阅侧 IDOR，独立任务处理。

        Args:
            turn_id: 由 start_turn() 返回的 ID。
            user_id: 可选，发起"立即回答"的当前登录用户 id。本地命中走 _turn_belongs_to；
                     跨 worker（本地 miss）回落 Redis owner key 比对（防 B 用户拿 A 的
                     turn_id 触发 A 的对话提前作答，Turn IDOR）。

        Returns:
            True  — 归属校验通过且信号已投递（本进程 ev.set / Redis 标志已置；跨 worker
                    为 best-effort，见上方残窗）
            False — turn 不存在/已结束（含 300s 回放窗内的死 turn），归属不符，或无 Redis
        """
        execution = self._executions.get(turn_id)
        # 归属校验（本进程持有 turn 时）：不符即 False，不得落入跨 worker 路径。
        # 未提供 user_id（内部/旧调用方）时 _turn_belongs_to 一律放行，保持对直接注入
        # event 的内部测试兼容。
        if execution is not None and not self._turn_belongs_to(execution, user_id):
            logger.warning(
                "TurnRuntime: request_answer_now turn_id=%s 归属不符", turn_id
            )
            return False
        ev = self._answer_now_events.get(turn_id)
        if ev is not None:
            # 同 worker 快路径：get 与 set 之间无 await，对 finally 的 pop 原子——
            # 不写 Redis 标志（watcher 见 ev 已置自行退出，不留垃圾 key）
            ev.set()
            log_flow("turn.answer_now", turn_id=turn_id)
            logger.info("TurnRuntime: answer_now requested for turn_id=%s", turn_id)
            return True
        if execution is not None:
            # turn 已结束（ev 已随 _run_turn finally 清）但 execution 还在回放 TTL 窗内：
            # 死 turn，必须 False（见 docstring 边界），不得落入下方 Redis 路径
            logger.info(
                "TurnRuntime: request_answer_now turn_id=%s turn 已结束（execution 回放窗内）",
                turn_id,
            )
            return False
        # 本地无 execution：跨 worker（或已彻底过期）。owner key 校验与 submit_user_reply
        # 共用 _cross_worker_owner_ok（跨 worker IDOR 比对口径单一真相源）。
        if not await self._cross_worker_owner_ok(turn_id, user_id, "request_answer_now"):
            return False
        ok = await reply_channel.set_answer_now(turn_id)
        if ok:
            log_flow("turn.answer_now", turn_id=turn_id, cross_worker=True)
            logger.info(
                "TurnRuntime: answer_now flag (cross-worker) set for turn_id=%s", turn_id
            )
        return ok

    @staticmethod
    def _turn_belongs_to(execution: _TurnExecution, user_id: str | None) -> bool:
        """turn 归属判定：未提供 user_id 时（旧调用方/内部用）一律放行，保持兼容；
        提供时要求 turn.context.user_id 严格相等。归属不符即视为"turn 不属于你"。"""
        if user_id is None:
            return True
        return str(execution.context.user_id or "") == str(user_id)

    @staticmethod
    async def _cross_worker_owner_ok(turn_id: str, user_id: str | None, action: str) -> bool:
        """跨 worker 归属判定（submit_user_reply / request_answer_now 共用）：本地
        _executions miss 时回落 Redis owner key（ca:turn:owner:{turn_id}）比对——与
        _turn_belongs_to 构成归属校验的「跨 worker 半边」，比对口径（str 强转 + 空
        owner 拒绝）单一真相源。未提供 user_id 一律放行，口径与 _turn_belongs_to 一致；
        action 仅用于日志文案。"""
        if user_id is None:
            return True
        owner = await reply_channel.get_turn_owner(turn_id)
        if not owner or str(owner) != str(user_id):
            logger.warning(
                "TurnRuntime: %s turn_id=%s 跨 worker 归属不符/不存在", action, turn_id
            )
            return False
        return True

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _run_turn(self, execution: _TurnExecution) -> None:
        """后台任务：orchestrator → fan-out → EventBus。"""
        from core.orchestrator import get_orchestrator
        from events.event_bus import CapabilityCompleteEvent, get_event_bus
        from core.observability.langsmith_trace import trace_context

        orchestrator = get_orchestrator()
        answer_parts: list[str] = []
        # 取消轮的兜底正文来源：TOKEN 流式增量（ANSWER 只在 loop 正常收尾才发，
        # 取消时 answer_parts 多半为空，部分输出只能从 token 流拼接）。
        token_parts: list[str] = []
        result_text_parts: list[str] = []
        result_metadata: dict = {}
        result_questions: list | None = None  # quiz RESULT 的原始题目列表（落库 metadata.quiz 用）
        done_metadata: dict = {}  # loop/quiz 收尾 done 事件的 metadata（usage/tools_used/...）
        turn_cancelled = False
        turn_error: str | None = None
        # 错误信封五元组的其余四员（turn_error 是 message）：由 ERROR 记账点或
        # generator 级 except 赋值，供 status/落盘/turn_errors 表消费
        turn_error_code = ""
        turn_error_ref = ""
        turn_error_exc_type = ""
        turn_error_exc_head = ""
        seq = 0
        first_event_logged = False
        t0 = time.monotonic()
        answer_now_watcher: asyncio.Task | None = None

        # 顶层 turn trace：下游所有 @traceable 工具/RAG 与 wrap_openai 的 LLM run
        # 通过 langsmith contextvars run tree 自动成为本 turn 的子 run。
        # 回合前裁剪（plan_turn/ContextBuilder）发生在 start_turn，早于本 trace 开始，故把
        # 结果塞进 metadata 直接挂在 turn 根 run 上，而非等它自然成为子 run（等不到）。
        async with trace_context(
            name="turn",
            metadata={
                "turn_id": execution.turn_id,
                "user_id": str(execution.context.user_id or ""),
                "course_id": str(execution.context.course_id or ""),
                "mode": str(execution.context.mode or ""),
                **{f"pretrim_{k}": v for k, v in (execution.context.metadata.get("_ctx_pretrim") or {}).items()},
            },
            tags=[f"mode:{execution.context.mode or 'chat'}"],
        ):
            # 回合前裁剪补一条可展开的 context.trim.pretrim 子 run（与轮内 context.trim.in_turn
            # 同形态）：裁剪发生在 start_turn，早于本 trace 开始，此前只把计数拍扁成 turn 根 run
            # metadata 的 pretrim_* 键——没有独立 run、更没有被砍正文，容易被忽略。dropped>0
            # 才记，零裁剪回合不产生噪声 run。
            _pre = execution.context.metadata.get("_ctx_pretrim") or {}
            if _pre.get("dropped"):
                from core.observability.langsmith_trace import record_context_trim
                record_context_trim(
                    stage="pretrim",
                    coordinator_enabled=bool(_pre.get("coordinator_enabled")),
                    iteration=-1,  # 回合前裁剪，不在任何轮内
                    dropped_messages=int(_pre.get("dropped") or 0),
                    summary_added=bool(_pre.get("carry_forward_added")),
                    dropped_preview="\n---\n".join(
                        execution.context.metadata.get("_ctx_pretrim_dropped_previews") or []
                    ),
                    summary_text=execution.context.metadata.get("_ctx_pretrim_summary_preview"),
                )
            try:
                # answer_now 跨 worker watcher：持有 turn 的 worker 轮询 Redis 标志位，
                # 命中置本地 ev（loop.py 的 is_answer_now_requested 回调零改动）。
                # 必须放 try 首句：任务在首步前被取消时协程体不执行、finally 不运行——
                # 在 start_turn 里 spawn 的话该场景下 watcher 将永久泄漏；放这里则
                # 「协程体执行过 ⇔ finally 必达 ⇔ watcher 必被 cancel」。无 Redis 不
                # 创建（memory:// 下跨 worker 投递本就不达，白轮询）。
                _answer_now_ev = self._answer_now_events.get(execution.turn_id)
                if _answer_now_ev is not None and reply_channel.redis_available():
                    answer_now_watcher = asyncio.create_task(
                        self._watch_answer_now_flag(execution.turn_id, _answer_now_ev),
                        name=f"ansnow-{execution.turn_id[:8]}",
                    )
                async for event in orchestrator.handle(execution.context):
                    # 记录 seq（供断线重连回放）
                    execution.events.append((seq, event))
                    seq += 1

                    if not first_event_logged:
                        log_flow("turn.first_event", turn_id=execution.turn_id,
                                 elapsed_ms=int((time.monotonic() - t0) * 1000),
                                 event_type=event.type.value if hasattr(event.type, "value") else str(event.type))
                        first_event_logged = True

                    # 收集最终答案文本
                    if event.type == StreamEventType.ANSWER:
                        answer_parts.append(str(event.payload.get("content") or ""))

                    # 收集流式增量（取消轮落盘部分输出用；正常轮 ANSWER 已含全文，不用它）
                    if event.type == StreamEventType.TOKEN:
                        token_parts.append(str(event.payload.get("content") or ""))

                    # 收集 result 产物（quiz/research 等能力不产 ANSWER、以 result 收尾）：
                    # 抽出可写入文本（research 报告 / quiz 题目），否则这两类能力永远不发
                    # CapabilityCompleteEvent，L3 outbox、巩固触发、L2 压缩全不跑。
                    if event.type == StreamEventType.RESULT:
                        report = event.payload.get("report")
                        if isinstance(report, str) and report.strip():
                            result_text_parts.append(report)
                        questions = event.payload.get("questions")
                        if isinstance(questions, list):
                            result_text_parts.append(
                                json.dumps(questions, ensure_ascii=False)
                            )
                            result_questions = questions
                        meta = event.payload.get("metadata")
                        if isinstance(meta, dict):
                            result_metadata.update(meta)

                    # 收集收尾 done 事件的 metadata（usage/tools_used/truncated/mode），
                    # 供回合落盘对齐前端此前写入的 metadata 集合。
                    if event.type == StreamEventType.DONE:
                        _dm = event.payload.get("metadata")
                        if isinstance(_dm, dict):
                            done_metadata.update(_dm)

                    # 错误信封记账点（单一）：orchestrator/能力/loop/本层四边界产生的
                    # 错误最终都以 ERROR 事件流经此处——回写五元组供 finally 统一
                    # status/指标/落库，天然只记一次。首写为准：多 ERROR 事件
                    # （loop 发完 pipeline 又抛）不双计。
                    if event.type == StreamEventType.ERROR and turn_error is None:
                        turn_error = str(event.payload.get("message") or "")
                        turn_error_code = str(event.payload.get("code") or "internal")
                        turn_error_ref = str(event.payload.get("ref") or "")
                        turn_error_exc_type = str(event.payload.get("exc_type") or "")
                        turn_error_exc_head = str(event.payload.get("exc_head") or "")

                    # fan-out 给所有 subscribe_turn 订阅者
                    await execution.bus.emit(event)

            except asyncio.CancelledError:
                turn_cancelled = True
                log_flow("turn.cancel", turn_id=execution.turn_id,
                         elapsed_ms=int((time.monotonic() - t0) * 1000), events=seq)
                # 取消也走信封（code=cancelled；ref 空串时 bus.error 按协议省略键）
                from core.observability.errors import error_ref

                await execution.bus.error(
                    "turn 被取消", source="turn_runtime",
                    code="cancelled", ref=error_ref(),
                )
            except Exception as exc:
                # 统一错误信封：分类（code）+ 中文文案 + ref；堆栈由 emit_error_event
                # 现场记录（turn_id 已在 contextvars，ContextFilter 注入每行日志）
                log_flow("turn.error", turn_id=execution.turn_id,
                         elapsed_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                from core.observability.errors import emit_error_event

                envelope = await emit_error_event(
                    execution.bus, exc, source="turn_runtime",
                )
                turn_error = envelope["message"]
                turn_error_code = envelope["code"]
                turn_error_ref = envelope["ref"]
                turn_error_exc_type = envelope["exc_type"]
                turn_error_exc_head = envelope["exc_head"]
            finally:
                execution.finished_at = time.monotonic()
                elapsed_total = int((execution.finished_at - t0) * 1000)
                agent_output = "".join(answer_parts)
                # quiz/research 等无 ANSWER 的能力：以 result 产物文本替代 agent_output，
                # 让 CapabilityCompleteEvent（记忆三层）对这些能力也触发。
                if not agent_output and result_text_parts:
                    agent_output = "\n\n".join(result_text_parts)
                # 错误/取消轮不再伪装成 empty/ok——失败率对 Prometheus 可见
                #（SSE 恒 200，HTTP 指标对本链路天然失明）
                if turn_error is not None:
                    status = "error"
                elif turn_cancelled:
                    status = "cancelled"
                else:
                    status = "ok" if agent_output else "empty"
                log_flow("turn.complete", turn_id=execution.turn_id,
                         elapsed_ms=elapsed_total, events=seq,
                         status=status, error_code=turn_error_code or None,
                         error_exc_type=turn_error_exc_type or None,
                         error_exc=turn_error_exc_head or None,
                         answer_chars=len(agent_output),
                         llm_usage=execution.context.metadata.get("llm_usage"),
                         llm_cost_usd=execution.context.metadata.get("llm_cost_usd"))
                observe_turn(
                    mode=execution.context.mode or "chat",
                    status=status,
                    elapsed_ms=elapsed_total,
                )
                if turn_error is not None:
                    observe_turn_error(
                        execution.context.mode or "chat",
                        turn_error_code or "internal",
                    )

                # 清理 reply_channel 进程内回退态（Redis 的 key 靠 TTL 自清理）。
                # turn 被取消时 waiter 的 BLPOP 由 CancelledError 解除，无需再 put(None)。
                reply_channel.cleanup(execution.turn_id)
                # 清理 answer_now 信号（turn 已结束，后续 request_answer_now 返回 False）
                self._answer_now_events.pop(execution.turn_id, None)
                # 取消 answer_now 跨 worker watcher。cancel 是同步调用且位于 finally 的
                # 无 await 前缀段——即使 finally 本身随后被二次取消（见下方 persist 的
                # shield 说明），cancel 也已发出；watcher 在下一个 await 点
                # （≤_ANSWER_NOW_POLL_S）退出并自行吞 CancelledError。cleanup → pop →
                # cancel 三条同步语句间无 await，不存在「pop 后、cancel 前插入 request」
                # 的窗口。
                if answer_now_watcher is not None:
                    answer_now_watcher.cancel()  # 对已完成的 task 是 no-op

                # 服务端权威回合日志：落 user/assistant 两行（幂等 by turn_id，含取消/错误轮）。
                # 必须先于 bus.close()——API 层（chat.py SSE finally / run.py 尾部）在订阅流
                # 一结束就 cancel_turn，close 之后才 persist 的话 cancel 会注入落盘的 DB
                # await，落盘整体失效。shield 再防一层：执行期间的 cancel 注入（用户停止/
                # 断流的二次 cancel）只打断外层收尾（publish/cooldown——取消语义下本就不跑），
                # 内层落盘保证跑完；捕下注入继续走 close，订阅者不悬挂。
                try:
                    await asyncio.shield(_persist_turn_rows(
                        execution,
                        agent_output=agent_output,
                        # token 拼接只在取消轮被读（正常轮 ANSWER 已含全文），不白付 O(全文) join
                        token_text="".join(token_parts) if turn_cancelled else "",
                        cancelled=turn_cancelled,
                        error_text=turn_error,
                        error_code=turn_error_code,
                        error_ref=turn_error_ref,
                        done_metadata=done_metadata,
                        result_metadata=result_metadata,
                        result_questions=result_questions,
                    ))
                except asyncio.CancelledError:
                    pass  # 落盘在后台继续；继续收尾（close bus 释放订阅者）
                # 失败 turn 的运维流水（turn_errors）：与叙事行独立 best-effort，互不拖累
                if turn_error is not None:
                    try:
                        await asyncio.shield(_persist_turn_error(
                            execution,
                            error_code=turn_error_code,
                            error_ref=turn_error_ref,
                            message=turn_error,
                            exc_type=turn_error_exc_type,
                            exc_head=turn_error_exc_head,
                        ))
                    except asyncio.CancelledError:
                        pass
                if not execution.bus._closed:
                    await execution.bus.close()

                # 发布 CAPABILITY_COMPLETE 事件（记忆更新等后台任务订阅）
                if agent_output:
                    try:
                        await get_event_bus().publish(CapabilityCompleteEvent(
                            turn_id=execution.turn_id,
                            session_id=execution.context.session_id,
                            user_id=execution.context.user_id,
                            course_id=execution.context.course_id,
                            mode=execution.context.mode or "chat",
                            user_message=execution.context.user_message,
                            agent_output=agent_output,
                            metadata=result_metadata or {},
                        ))
                    except Exception:
                        logger.warning("TurnRuntime: EventBus publish failed", exc_info=True)
                    # 学情门控冷却在回合成功后才落盘（拼上下文≠成功；模型失败不烧 24h）。
                    # 按 (user, course, topic) upsert，不依赖 mastery 行是否已有（C2 复发
                    # 证据来自观测日志，可能还没折叠出行）。
                    _pending = (execution.context.metadata or {}).get(
                        "_pending_stitch_cooldown"
                    ) or {}
                    if _pending and execution.context.user_id and execution.context.course_id:
                        try:
                            from core.db.database import AsyncSessionLocal
                            from core.memory.proactive import commit_stitch_cooldown

                            async with AsyncSessionLocal() as db:
                                for topic_id in _pending:
                                    await commit_stitch_cooldown(
                                        db,
                                        str(execution.context.user_id),
                                        str(execution.context.course_id),
                                        str(topic_id),
                                    )
                        except Exception:
                            logger.warning(
                                "TurnRuntime: stitch cooldown commit failed turn_id=%s",
                                execution.turn_id, exc_info=True,
                            )

                # TTL 后清理内存
                asyncio.create_task(
                    self._schedule_cleanup(execution.turn_id),
                    name=f"cleanup-{execution.turn_id[:8]}",
                )

    async def _schedule_cleanup(self, turn_id: str) -> None:
        """在 TTL 后从内存中移除 turn 状态。"""
        await asyncio.sleep(_EXECUTION_TTL)
        async with self._lock:
            self._executions.pop(turn_id, None)
        logger.debug("TurnRuntime: cleaned up turn_id=%s", turn_id)

    async def _watch_answer_now_flag(self, turn_id: str, ev: asyncio.Event) -> None:
        """跨 worker answer_now 信号 watcher：轮询 Redis 标志位，命中即置本地 ev。

        退出三途：① 标志命中（置 ev 后 return）② ev 已被本进程直连路径置位（while
        条件不成立）③ _run_turn finally 的 cancel——turn 正常结束但信号从未命中时
        watcher 不会自己退出，cancel 是唯一退路，故吞 CancelledError（cancel 即正常
        退出，不是异常）。check 异常按未触发处理，下一轮再读（Redis 抖动不杀 watcher）。
        """
        try:
            while not ev.is_set():
                if await reply_channel.check_answer_now(turn_id):
                    ev.set()
                    log_flow("turn.answer_now_cross_worker", turn_id=turn_id)
                    logger.info(
                        "TurnRuntime: answer_now flag consumed (cross-worker) turn_id=%s",
                        turn_id,
                    )
                    return
                await asyncio.sleep(_ANSWER_NOW_POLL_S)
        except asyncio.CancelledError:
            return


# ---- 全局单例 ----

_manager: TurnRuntimeManager | None = None


def get_turn_runtime_manager() -> TurnRuntimeManager:
    """返回全局 TurnRuntimeManager 单例（懒加载）。"""
    global _manager
    if _manager is None:
        _manager = TurnRuntimeManager()
    return _manager
