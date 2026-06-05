from src.nodes.intent_parser import _normalize_llm_intent, parse_intent


def test_intent_parser_counts_partner_and_child_plus_self() -> None:
    intent = parse_intent(
        "下午带5岁孩子和伴侣出去，孩子要安全有趣，伴侣最近在控热，不想排队太久。"
    )

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3


def test_llm_normalization_corrects_partner_and_child_plus_self_count() -> None:
    user_input = (
        "明天一天安排亲子活动，孩子5岁要安全有趣，伴侣控热，不想排队太久。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "family",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "partner", "needs": ["comfortable"]},
            {"role": "child", "age": 5, "needs": ["儿童友好"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 3
