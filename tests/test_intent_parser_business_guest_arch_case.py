from src.nodes.intent_parser import parse_intent


def test_business_guest_request_counts_hosts_and_preserves_after_meeting_start() -> None:
    request = (
        "明天下午两位客户在陆家嘴开完会，13点以后我和同事想带他们在上海看看有内容的地方，"
        "最好是博物馆或展览，有讲解更好；中间想顺手买点不占地方的上海伴手礼，"
        "18:30左右吃一顿清淡点、方便聊天的本帮或江浙菜，其中一位客户不吃猪肉。"
        "20:30前必须把他们送到虹桥火车站，人均300左右，不想去网红排队店。"
        "展览和晚餐能预约就预约，伴手礼如果只能给建议也要说清楚。"
    )

    intent = parse_intent(request)

    assert intent["people_count"] == 4
    assert intent["time"]["window"] == "tomorrow_afternoon"
    assert intent["time"]["start_time"] == "13:00"
    assert intent["time"]["end_time"] == "20:30"
    assert intent["budget"] == {
        "amount": 300,
        "type": "per_person",
        "sensitivity": "high",
    }
    assert "不含猪肉" in intent["constraints"]["hard"]
    assert "不含猪肉" in intent["explicit_constraints"]["dietary"]


def test_dental_aftercare_business_request_counts_hosts_and_negates_loud_venues() -> None:
    request = (
        "今天我和同事陪两个外地客户，17:40他们在人民广场附近做完口腔处理出来，"
        "20:45前要送回南京东路酒店。想安排一个轻松的上海味晚饭，"
        "再有一个坐着聊项目的地方，别喝酒、别太辣、别太硬，"
        "也不要需要大声说话的演出或KTV。人均250以内，最好能订位；"
        "如果你不能直接约牙科或酒店接送，就不要把它写成已安排。"
    )

    intent = parse_intent(request)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 4
    assert intent["time"]["start_time"] == "17:40"
    assert intent["time"]["end_time"] == "20:45"
    assert intent["budget"] == {
        "amount": 250,
        "type": "per_person",
        "sensitivity": "high",
    }
    assert {"不辣", "无酒精", "软食"} <= set(intent["constraints"]["hard"])
    assert "KTV欢唱" in intent["constraints"]["avoid"]
    assert "KTV欢唱" not in intent["constraints"]["soft"]
