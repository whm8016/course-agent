"""QuizPipeline 解析 + schema 校验单测（Phase 3）。

验证 _parse_templates / _parse_question 的 6 题型规范化、choice options ABCD 过滤、
count 截断、空 topic 丢弃、question_type 继承 template。
"""
from core.question.pipeline import _parse_question, _parse_templates


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
