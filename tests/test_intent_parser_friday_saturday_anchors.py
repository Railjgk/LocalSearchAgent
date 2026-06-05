from src.nodes.intent_parser import parse_intent


def test_intent_parser_preserves_friday_evening_to_saturday_deadline() -> None:
    intent = parse_intent(
        "帮我排一个周五晚上到周六下午的安排：周五18点从世纪大道附近出发，"
        "先尽量找个靠谱口腔门诊看看牙疼或至少能咨询，之后吃点软一点、低盐清淡的晚饭；"
        "周六上午想带爸爸看个安静点的展或博物馆，中午继续吃软一点的清淡饭，16点前结束。"
    )

    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "18:00"
    assert intent["time"]["end_time"] == "16:00"
    assert intent["time"]["duration_range"] == [22, 22]
