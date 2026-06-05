from src.nodes.intent_parser import _normalize_llm_intent, parse_intent


def test_intent_parser_counts_plural_adults_plus_child() -> None:
    user_input = (
        "下午想带5岁孩子在家附近找个地方活动一下，孩子需要安全有趣，"
        "我们大人希望找个能顺便吃点健康轻食的地方，不想排队太久。"
    )

    intent = parse_intent(user_input)

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3


def test_intent_parser_counts_separated_plural_adults_plus_child() -> None:
    user_input = (
        "想安排一天亲子行程，孩子5岁，下午我们更空闲，希望孩子玩得开心，"
        "大人能控制饮食，预算中等，尽量少换乘，不想排队太久。"
    )

    intent = parse_intent(user_input)

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3


def test_llm_normalization_corrects_plural_adults_plus_child_count() -> None:
    user_input = (
        "明天下午想带5岁孩子出去玩，孩子需要安全有趣的活动，"
        "我们大人希望找个健康轻食的地方吃饭。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "family",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "child", "age": 5, "needs": ["儿童友好"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 3


def test_llm_normalization_corrects_separated_plural_adults_plus_child_count() -> None:
    user_input = (
        "计划两天亲子游，孩子5岁，下午我们更空闲，希望孩子玩得开心，"
        "大人能控制饮食，预算中等，尽量少换乘，不想排队太久。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "family",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "child", "age": 5, "needs": ["儿童友好"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 3
