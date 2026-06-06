from src.memory.policy import apply_value_memory
from src.memory.storage import load_memory
from src.nodes.b_requirement_compiler import apply_b_requirement_contract
from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint
from src.nodes.intent_parser import constraints_from_intent, parse_intent
from src.nodes.scenario_planner import build_scenario_plan


def test_no_child_date_suppresses_child_and_diet_memory_overrides() -> None:
    text = (
        "今晚孩子去外婆家了，就我和老婆两个人，20点以后从静安寺附近出发，"
        "想吃一顿正常点的本帮菜或小酒馆，再看场电影或者找个安静清吧坐一下，"
        "23:30前回家。别再按上次带娃和减脂轻食那套给我排，"
        "老婆最近不想被提醒减肥；她不吃内脏，预算人均350左右。"
    )

    intent = parse_intent(text)

    assert intent["scene"] == "couple"
    assert intent["people_count"] == 2
    assert all(person["role"] != "child" for person in intent["people"])
    assert "亲子" not in intent["planning_preferences"]["activity_type"]
    assert "儿童友好" not in intent["constraints"]["hard"]
    assert "低卡" not in intent["planning_preferences"]["food_type"]
    assert "轻食" not in intent["planning_preferences"]["food_type"]

    constraints = constraints_from_intent(intent)
    merged = apply_value_memory(constraints, load_memory("u001"))

    assert constraints["sequence_preference"] == "restaurant_then_activity"
    assert merged["child_age"] is None
    assert merged["mom_diet"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "low_calorie" not in merged["soft_tags"]
    assert "light_food" not in merged["soft_tags"]
    assert "family_care" not in merged["active_value_ids"]
    assert "health" not in merged["active_value_ids"]

    state = {"user_input": text, "intent": intent, "constraints": merged}
    scenario_plan = build_scenario_plan(state)
    merged["scenario_activities"] = scenario_plan["scenario_activities"]
    merged["scenario_facets"] = scenario_plan["scenario_facets"]
    contract_constraints, contract, _metadata = apply_b_requirement_contract(
        {**state, **scenario_plan, "constraints": merged},
        constraints=merged,
    )

    assert "亲子" not in scenario_plan["scenario_facets"]["relationship"]
    assert "儿童友好" not in scenario_plan["scenario_facets"]["relationship"]
    assert "child_friendly_activity" not in contract["hard_requirements"]
    assert "child_friendly_activity" not in contract_constraints[
        "b_requirement_contract"
    ]["hard_requirements"]

    blueprint = build_b_itinerary_blueprint(state, constraints=merged)
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert roles[0] in {"restaurant_dinner", "restaurant_specific"}
    assert "cinema" in roles[1:]
    assert "bar" in roles[1:]
    assert "family_activity" not in roles
    assert "family_indoor_play" not in roles
