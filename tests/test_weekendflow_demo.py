from src.graph import get_graph
from src.nodes.intent_parser import constraints_from_intent, parse_intent
from src.nodes.memory_manager import apply_value_memory, load_memory
from src.nodes.scenario_planner import build_scenario_plan


def test_intent_parser_extracts_family_constraints() -> None:
    intent = parse_intent(
        "今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远"
    )

    assert intent["time"]["window"] == "today_afternoon"
    assert intent["scene"] == "family"
    assert intent["location"]["max_distance_km"] == 8.0
    assert "kid_friendly" in intent["constraints"]["hard"]
    assert "low_calorie" in intent["constraints"]["soft"]
    assert "budget" in intent["missing_slots"]


def test_intent_parser_handles_message_input_and_friends_scene() -> None:
    intent = parse_intent(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "下午和朋友出去玩，4个人"},
        ]
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 4
    assert "group_activity" in intent["planning_preferences"]["activity_type"]


def test_graph_accepts_messages_when_user_input_missing() -> None:
    graph = get_graph()
    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "下午和朋友出去玩，4个人"},
            ],
        }
    )

    assert result["user_input"] == "下午和朋友出去玩，4个人"
    assert result["scene_type"] == "friends"
    assert result["constraints"]["people_count"] == 4


def test_memory_does_not_apply_family_defaults_to_friends_request() -> None:
    intent = parse_intent("下午和朋友出去玩，4个人")
    constraints = constraints_from_intent(intent)
    merged = apply_value_memory(constraints, load_memory("u001"))

    assert merged["scene"] == "friends"
    assert merged["people_count"] == 4
    assert merged["child_age"] is None
    assert merged["mom_diet"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "low_calorie" not in merged["soft_tags"]
    assert "family_care" not in merged["active_value_ids"]
    assert "health" not in merged["active_value_ids"]


def test_scenario_planner_outputs_a_to_b_handoff() -> None:
    intent = parse_intent(
        "今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远"
    )
    constraints = apply_value_memory(constraints_from_intent(intent), load_memory("u001"))
    scenario_plan = build_scenario_plan(
        {
            "intent": intent,
            "constraints": constraints,
        }
    )

    assert scenario_plan["scene_type"] == "family"
    assert "亲子乐园" in scenario_plan["scenario_activities"]
    assert "轻食餐厅" in scenario_plan["scenario_activities"]
    assert scenario_plan["scenario_template"]["poi_mix"] == ["activity", "restaurant"]
    assert scenario_plan["route_pattern_hints"]["should_search"] is True


def test_weekendflow_a_stage_outputs_intent_and_value_memory() -> None:
    graph = get_graph()
    result = graph.invoke(
        {
            "user_id": "u001",
            "user_input": (
                "今天下午想和老婆孩子出去玩几个小时，"
                "别离家太远，孩子5岁，老婆最近在减肥。"
            ),
        }
    )

    assert result["scene_type"] == "family"
    assert result["intent"]["location"]["distance_preference"] == "nearby"
    assert result["constraints"]["max_queue_time_min"] == 15
    assert result["constraints"]["value_weights"]["family_care"] == 0.92
    assert result["memory"]["companion_profile"]["child"]["age"] == 5
    assert result["memory"]["value_profile"][0]["planning_effect"]
    assert result["constraints"]["memory_policy"] == "explicit_current_input_first"
    assert result["scenario_activities"]
    assert result["constraints"]["scenario_activities"] == result["scenario_activities"]
    assert result["short_term_memory"]
