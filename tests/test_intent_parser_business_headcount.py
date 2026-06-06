from src.nodes.intent_parser import _normalize_llm_intent, parse_intent


BUSINESS_RECEPTION_REQUEST = (
    "我这周接待两位外地客户，两天都在上海。第一天14:00从浦东机场附近开始，"
    "想轻松看点上海特色、晚上吃个安静体面的饭；第二天上午客户想找个靠谱旅行社"
    "咨询日本签证，下午买点上海伴手礼，17:00前我得把他们送到虹桥站。"
    "我们一共3个人，其中一位素食，两位都不吃辣；总预算1800以内。"
)


def test_intent_parser_prefers_explicit_total_headcount_over_customer_count() -> None:
    intent = parse_intent(BUSINESS_RECEPTION_REQUEST)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert intent["people"] == [
        {"role": "self", "needs": []},
        {"role": "friends", "count": 2, "needs": ["group_friendly", "social"]},
    ]


def test_llm_normalization_prefers_explicit_total_headcount_over_customer_count() -> None:
    mock_intent = parse_intent(BUSINESS_RECEPTION_REQUEST)
    raw_intent = {
        "scene": "solo",
        "people_count": 2,
        "people": [{"role": "self", "needs": []}],
    }

    normalized = _normalize_llm_intent(
        raw_intent,
        mock_intent,
        BUSINESS_RECEPTION_REQUEST,
    )

    assert normalized["scene"] == "friends"
    assert normalized["people_count"] == 3
