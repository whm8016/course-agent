"""生产问答回流脚本的纯逻辑测试（Phase 6）。

只测碰 DB 之外的纯函数：classify_answer / pair_user_assistant / to_candidate。
fetch_qa_pairs 依赖真实库（集成路径），不在此单测。
"""


# ---------------------------------------------------------------------------
# classify_answer: 启发式初筛
# ---------------------------------------------------------------------------
def test_classify_answer_refusal():
    from scripts.eval_rag.production_feedback import classify_answer
    suspect, reason = classify_answer("抱歉，我没有找到相关的资料。")
    assert suspect and "refusal" in reason


def test_classify_answer_too_short():
    from scripts.eval_rag.production_feedback import classify_answer
    suspect, reason = classify_answer("好的")
    assert suspect and "too_short" in reason


def test_classify_answer_empty():
    from scripts.eval_rag.production_feedback import classify_answer
    suspect, reason = classify_answer("")
    assert suspect and reason == "answer_empty"


def test_classify_answer_normal_ok():
    from scripts.eval_rag.production_feedback import classify_answer
    # 正常充实的回答不应被判可疑
    suspect, _ = classify_answer(
        "基尔霍夫电流定律（KCL）指出：流入任一节点的电流之和等于流出该节点的电流之和，"
        "这是电路分析的基本约束之一。"
    )
    assert not suspect


# ---------------------------------------------------------------------------
# pair_user_assistant: 配对逻辑
# ---------------------------------------------------------------------------
def _msg(mid, role, content, ts, course="c"):
    return {"id": mid, "role": role, "content": content, "created_at": ts, "course_id": course}


def test_pair_basic():
    from scripts.eval_rag.production_feedback import pair_user_assistant
    by_session = {
        "s1": [
            _msg("m1", "user", "什么是KCL？", 1.0),
            _msg("m2", "assistant", "KCL是基尔霍夫电流定律。", 2.0),
            _msg("m3", "user", "KVL呢？", 3.0),
            _msg("m4", "assistant", "KVL是基尔霍夫电压定律。", 4.0),
        ],
    }
    pairs = pair_user_assistant(by_session)
    assert len(pairs) == 2
    assert pairs[0]["question"] == "什么是KCL？"
    assert pairs[0]["answer"] == "KCL是基尔霍夫电流定律。"
    assert pairs[1]["q_msg_id"] == "m3"


def test_pair_skips_user_without_followup():
    """user 后无 assistant 跟随（被中断）→ 不配对。"""
    from scripts.eval_rag.production_feedback import pair_user_assistant
    by_session = {
        "s1": [_msg("m1", "user", "孤立问题？", 1.0)],
        "s2": [_msg("m2", "user", "q", 2.0), _msg("m3", "assistant", "a", 3.0)],
    }
    pairs = pair_user_assistant(by_session)
    assert len(pairs) == 1
    assert pairs[0]["q_msg_id"] == "m2"


def test_pair_cutoff_filters_old():
    """cutoff 只保留 created_at >= cutoff 的 user 提问。"""
    from scripts.eval_rag.production_feedback import pair_user_assistant
    by_session = {
        "s1": [
            _msg("m1", "user", "old", 1.0),
            _msg("m2", "assistant", "a", 2.0),
            _msg("m3", "user", "new", 100.0),
            _msg("m4", "assistant", "b", 101.0),
        ],
    }
    pairs = pair_user_assistant(by_session, cutoff_ts=50.0)
    assert len(pairs) == 1 and pairs[0]["question"] == "new"


# ---------------------------------------------------------------------------
# to_candidate: 导出条目
# ---------------------------------------------------------------------------
def test_to_candidate_marks_suspect_and_no_ground_truth():
    from scripts.eval_rag.production_feedback import to_candidate
    pair = {
        "question": "q", "answer": "我不知道",
        "course_id": "c", "session_id": "s", "created_at": 1.0, "q_msg_id": "m1",
    }
    c = to_candidate(pair)
    assert c["id"] == "prod_m1"
    assert c["source"] == "production"
    assert c["ground_truth"] is None  # 真实问题无标准答案
    assert c["suspect_low_quality"] is True
    assert c["suspect_reason"].startswith("refusal")
