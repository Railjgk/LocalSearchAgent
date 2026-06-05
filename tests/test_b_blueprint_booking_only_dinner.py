from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint


def test_booking_preference_does_not_create_extra_dinner_node():
    text = (
        "明天上午从中山公园附近带金毛去洗护修毛，最好能预约安静点的宠物店；"
        "我等的时候想在附近找个能办公的咖啡馆，11:30朋友过来吃轻食早午餐。"
        "朋友乳糖不耐，15:00前要回家，总预算500以内，能订座的就订座。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "start_time": "11:30",
            "end_time": "15:00",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert any(role in roles for role in ("restaurant_lunch", "restaurant_specific"))
    assert "restaurant_dinner" not in roles
