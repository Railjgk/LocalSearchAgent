from src.nodes.intent_parser import parse_intent


def test_intent_parser_counts_headcount_followed_by_clock_time() -> None:
    intent = parse_intent(
        "老板临时说今晚给同事补个生日，6个人17:20在人民广场集合。"
        "想先取一个小蛋糕和一束花，18:30前到桌游。"
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 6
    assert next(item for item in intent["people"] if item["role"] == "friends")[
        "count"
    ] == 5
