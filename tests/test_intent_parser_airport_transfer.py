from src.nodes.intent_parser import parse_intent


def test_airport_transfer_exhibit_request_preserves_arrival_and_departure_anchors() -> None:
    intent = parse_intent(
        "后天朋友13:10到虹桥站，晚上21:20从浦东机场飞走，中间想带他快速看看上海。"
        "我们两个人、行李箱比较大，别排太散：先找个能坐下休息的咖啡馆或给出靠谱寄存建议，"
        "再去上海博物馆那个“古埃及文明大展”或同类有代表性的展览，晚饭要清真或至少不含猪肉，"
        "20:00前必须开始往浦东机场走。总预算700以内，不想现场碰运气排长队。"
    )

    assert intent["time"]["window"] == "day_after_tomorrow"
    assert intent["time"]["start_time"] == "13:10"
    assert intent["time"]["end_time"] == "20:00"
    assert intent["location"]["origin"] == "虹桥站"
    assert "浦东机场飞" not in intent["location"]["origin"]
