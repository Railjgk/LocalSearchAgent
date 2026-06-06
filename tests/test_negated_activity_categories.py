from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint
from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_soccer_request_blocks_forbidden_family_and_museum_fallbacks() -> None:
    text = (
        "周日下午想带8岁儿子在闵行莘庄附近试一次少儿足球试听课，两位大人陪同，"
        "14:00左右出发，19:00前回到莘庄。孩子最近有点过敏性咳嗽，所以训练别超过90分钟，"
        "强度别太猛；这次不要再给我排亲子乐园或博物馆。17:30以后在附近吃个不辣、少油的晚饭。"
    )

    intent = parse_intent(text)

    assert "亲子" in intent["constraints"]["avoid"]
    assert "博物馆展览" in intent["constraints"]["avoid"]
    assert "亲子" not in intent["planning_preferences"]["activity_type"]
    assert "博物馆展览" not in intent["planning_preferences"]["activity_type"]

    constraints = constraints_from_intent(intent)
    blueprint = build_b_itinerary_blueprint(
        {"user_input": text, "scene_type": intent["scene"], "constraints": constraints},
        constraints=constraints,
    )
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "sports_training" in roles
    assert "family_activity" not in roles
    assert "family_indoor_play" not in roles
    assert "exhibition" not in roles
