from src.nodes.intent_parser import _normalize_llm_intent, parse_intent


def test_intent_parser_preserves_separated_weekend_two_day_anchors() -> None:
    user_input = (
        "这个周末帮我们排两天，但不用订住宿，周六晚上住外婆家。"
        "周六13:00从中山公园附近开车出发，想给8岁孩子找个少儿足球体验课；"
        "周日上午带猫去做个基础体检，顺便买点猫粮，"
        "午饭后15:00前回到中山公园附近。总预算900以内。"
    )

    intent = parse_intent(user_input)

    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "13:00"
    assert intent["time"]["end_time"] == "15:00"
    assert intent["time"]["duration_range"] == [26, 26]


def test_llm_normalization_keeps_separated_weekend_duration() -> None:
    user_input = (
        "这个周末帮我们排两天，但不用订住宿，周六晚上住外婆家。"
        "周六13:00从中山公园附近开车出发，想给8岁孩子找个少儿足球体验课；"
        "周日上午带猫去做个基础体检，午饭后15:00前回到中山公园附近。"
    )
    baseline = parse_intent(user_input)
    raw_intent = {
        "time": {
            "window": "weekend",
            "duration_range": [3, 6],
            "start_time": None,
            "end_time": None,
        }
    }

    normalized = _normalize_llm_intent(raw_intent, baseline, user_input)

    assert normalized["time"]["start_time"] == "13:00"
    assert normalized["time"]["end_time"] == "15:00"
    assert normalized["time"]["duration_range"] == [26, 26]


def test_intent_parser_preserves_period_only_weekend_departure_anchor() -> None:
    intent = parse_intent(
        "这个周末想和对象过纪念日，两天一夜，周六中午出发、"
        "周日16点前结束，自驾，会带一只小狗。"
    )

    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "12:00"
    assert intent["time"]["end_time"] == "16:00"
    assert intent["time"]["duration_range"] == [28, 28]
