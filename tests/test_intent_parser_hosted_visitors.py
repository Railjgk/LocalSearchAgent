from src.nodes.intent_parser import parse_intent


def test_intent_parser_counts_self_plus_two_visiting_classmates() -> None:
    intent = parse_intent(
        "周六有两个外地同学来上海玩一天，10:00在人民广场碰头，"
        "想看上海博物馆那个需要预约的大展，午饭吃上海特色。"
        "20:30前我要把他们送到虹桥火车站，人均300左右。"
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert next(item for item in intent["people"] if item["role"] == "friends")[
        "count"
    ] == 2
