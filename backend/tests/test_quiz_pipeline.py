"""QuizPipeline 解析 + schema 校验单测（Phase 3 + A2 判分闭环）。

验证 _parse_templates / _parse_question 的 6 题型规范化、choice options ABCD 过滤、
count 截断、空 topic 丢弃、question_type 继承 template。
A2：coding 题新字段（language/starter_code/test_cases）解析/清洗/截断，
_validate_question 对 coding 题要求 starter_code + ≥2 条 assert 用例。
"""
from core.question.pipeline import _parse_question, _parse_templates, _validate_question


def test_parse_templates_normalizes_and_drops_empty_topic():
    text = (
        '{"analysis":"x","templates":['
        '{"question_id":"q_1","topic":"导数定义","question_type":"Choice","difficulty":"Hard"},'
        '{"question_id":"q_2","topic":"积分","question_type":"weird","difficulty":"easy"},'
        '{"question_id":"q_3","topic":"","question_type":"written","difficulty":"medium"}'
        "]}"
    )
    templates = _parse_templates(text, count=5)
    assert len(templates) == 2  # 空 topic 丢弃
    assert templates[0]["question_type"] == "choice"   # Choice → choice
    assert templates[0]["difficulty"] == "hard"
    assert templates[1]["question_type"] == "written"  # 非法题型 → written


def test_parse_templates_count_cap():
    text = (
        '{"templates":['
        '{"topic":"a","question_type":"written","difficulty":"easy"},'
        '{"topic":"b","question_type":"written","difficulty":"easy"},'
        '{"topic":"c","question_type":"written","difficulty":"easy"}'
        "]}"
    )
    assert len(_parse_templates(text, count=2)) == 2


def test_parse_question_choice_filters_options():
    text = (
        '{"question_type":"choice","question":"1+1=?",'
        '"options":{"A":"1","B":"2","C":"3","D":"4","E":"5"},'
        '"correct_answer":"B","explanation":"1+1=2"}'
    )
    q = _parse_question(text, {"question_id": "q_1", "question_type": "choice", "difficulty": "easy", "topic": "加法"})
    assert q["question_type"] == "choice"
    assert q["options"] == {"A": "1", "B": "2", "C": "3", "D": "4"}  # E 被过滤
    assert q["correct_answer"] == "B"


def test_parse_question_concept_has_no_options():
    text = '{"question_type":"concept","question":"地球是圆的","correct_answer":"true","explanation":"近似球体"}'
    q = _parse_question(text, {"question_id": "q_1", "question_type": "concept", "difficulty": "easy", "topic": "地理"})
    assert q["question_type"] == "concept"
    assert q["options"] is None
    assert q["correct_answer"] == "true"


def test_parse_question_inherits_template_type_when_missing():
    text = '{"question":"解释熵增","correct_answer":"...","explanation":"..."}'
    q = _parse_question(text, {"question_id": "q_1", "question_type": "written", "difficulty": "medium", "topic": "热力学"})
    assert q["question_type"] == "written"


# ── A2：coding 题新字段 ──────────────────────────────────────────────────────


_CODING_TMPL = {"question_id": "q_1", "question_type": "coding", "difficulty": "easy", "topic": "python 基础"}


def test_parse_question_coding_extracts_new_fields():
    text = (
        '{"question_type":"coding","question":"实现两数相加",'
        '"correct_answer":"```python\\ndef add(a, b):\\n    return a + b\\n```",'
        '"explanation":"...",'
        '"starter_code":"def add(a, b):\\n",'
        '"test_cases":["assert add(1, 2) == 3", " assert add(0, 0) == 0 ", "", "print(1)"]}'
    )
    q = _parse_question(text, _CODING_TMPL)
    assert q["language"] == "python"
    # strip 口径：骨架首尾空白无意义，run_tests 拼接时自带 "\n"、前端 textarea 整段渲染
    assert q["starter_code"] == "def add(a, b):"
    # strip 去空 + 只留 assert 前缀（生成噪声条目直接滤掉，别让一条坏用例杀死整题）
    assert q["test_cases"] == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]


def test_parse_question_coding_caps_ten_cases():
    cases = ",".join(f'"assert f({i}) == {i}"' for i in range(12))
    text = (
        '{"question_type":"coding","question":"实现 f",'
        '"correct_answer":"```python\\ndef f(x):\\n    return x\\n```",'
        '"explanation":"...","starter_code":"def f(x):\\n","test_cases":[' + cases + "]}"
    )
    q = _parse_question(text, _CODING_TMPL)
    assert len(q["test_cases"]) == 10  # 钳制上限（防 LLM 失控用例串）


def test_parse_question_non_coding_gets_empty_defaults():
    """全题型扁平 schema 约定：非 coding 题也带三个新键（空默认），下游免条件分支。"""
    text = '{"question":"解释光合作用","correct_answer":"...","explanation":"..."}'
    q = _parse_question(text, {"question_id": "q_1", "question_type": "written", "difficulty": "easy", "topic": "生物"})
    assert q["language"] == ""
    assert q["starter_code"] == ""
    assert q["test_cases"] == []


def test_validate_question_coding_requires_starter_and_two_asserts():
    base = {
        "question": "实现 add",
        "question_type": "coding",
        "correct_answer": "```python\ndef add(a, b):\n    return a + b\n```",
        "explanation": "...",
    }
    ok_q = dict(base, starter_code="def add(a, b):\n", test_cases=["assert add(1, 1) == 2", "assert add(0, 0) == 0"])
    assert _validate_question(ok_q) is True

    assert _validate_question(dict(ok_q, test_cases=[])) is False  # 无用例判不了
    assert _validate_question(dict(ok_q, test_cases=["assert add(1, 1) == 2"])) is False  # 单条不足
    assert _validate_question(dict(ok_q, starter_code="")) is False  # 无骨架
    # 非 assert 前缀在 _parse 已滤——这里模拟绕过 parse 的直接构造仍要拒绝
    assert _validate_question(dict(ok_q, test_cases=["assert add(1, 1) == 2", "print(1)"])) is False


def test_validate_question_non_coding_unchanged():
    """回归：非 coding 题校验行为不变（新字段不参与）。"""
    assert _validate_question({
        "question": "解释熵增", "question_type": "written",
        "correct_answer": "...", "explanation": "...",
    }) is True
