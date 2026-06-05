from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_destination_city_takes_priority_over_current_city_context():
    intent = parse_intent(
        "\u76ee\u7684\u5730\u57ce\u5e02\uff1a\u9752\u5c9b\u3002"
        "\u7528\u6237\u539f\u59cb\u8f93\u5165\uff1a\u6211\u4eba\u5728\u4e0a\u6d77\uff0c"
        "\u4f46\u662f\u8fd9\u4e2a\u5468\u672b\u60f3\u5728\u9752\u5c9b"
        "\u8f7b\u677e\u73a9\u4e00\u5929\uff0c\u60f3\u5403\u6d77\u9c9c\uff0c\u522b\u592a\u7d2f"
    )
    constraints = constraints_from_intent(intent)

    assert constraints["city"] == "\u9752\u5c9b"
    assert constraints["time_window"] == "full_day"


def test_go_to_city_pattern_is_treated_as_trip_city():
    intent = parse_intent(
        "\u6211\u73b0\u5728\u5728\u4e0a\u6d77\uff0c"
        "\u5468\u672b\u53bb\u9752\u5c9b\u8fc7\u5468\u672b\uff0c"
        "\u60f3\u5403\u6d77\u9c9c"
    )
    constraints = constraints_from_intent(intent)

    assert constraints["city"] == "\u9752\u5c9b"
