from src.memory.policy import apply_value_memory
from src.memory.storage import load_memory
from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_birthday_request_rejects_low_calorie_limit_and_keeps_seafood_avoidance() -> None:
    text = (
        "周五晚上想给女朋友过生日，18点我从静安寺下班出发，22:30前结束。"
        "想先顺路取一束花或小蛋糕，再带她做个轻量的小惊喜，然后吃一顿正常的生日晚餐。"
        "她这次明确不想被低卡减脂限制，但海鲜不吃；总预算1200以内。"
    )

    intent = parse_intent(text)
    constraints = constraints_from_intent(intent)
    merged = apply_value_memory(constraints, load_memory("u001"))

    assert "低卡" not in intent["planning_preferences"]["food_type"]
    assert "轻食" not in intent["planning_preferences"]["food_type"]
    assert "低卡" not in intent["constraints"]["soft"]
    assert "轻食" not in intent["constraints"]["soft"]
    assert "过敏友好" in intent["constraints"]["hard"]
    assert "避开海鲜" in intent["constraints"]["hard"]
    assert merged["mom_diet"] is None
    assert "low_calorie" not in merged["soft_tags"]
    assert "light_food" not in merged["soft_tags"]
    assert "health" not in merged["active_value_ids"]
