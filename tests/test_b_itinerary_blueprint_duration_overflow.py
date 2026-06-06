from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint


def _minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def test_blueprint_does_not_severely_compress_overbooked_node_durations():
    text = (
        "周日下午我和对象带金毛出门，不带孩子。14:00以后从徐汇滨江附近自驾，"
        "想先找个宠物洗澡修毛的地方，洗护大概90分钟；我们等的时候在附近喝杯咖啡坐坐，"
        "17:30前必须接到狗，之后如果还有余力吃个能带狗的晚饭。"
    )
    state = {
        "user_input": text,
        "scene_type": "couple",
        "constraints": {
            "raw_text": text,
            "start_time": "14:00",
            "end_time": "17:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    slots = blueprint["time_skeleton"]["days"][0]["slots"]
    grooming = next(slot for slot in slots if slot["role"] == "pet_grooming")

    assert grooming["duration_min"] == 90
    assert _minutes(grooming["end_time"]) - _minutes(grooming["start_time"]) == 90
    assert grooming["time_window_overflow_min"] > 0
