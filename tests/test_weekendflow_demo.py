from src.graph import get_graph
from src.nodes.intent_parser import parse_intent


def test_intent_parser_extracts_family_constraints() -> None:
    intent = parse_intent("今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远")

    assert intent["time"]["window"] == "today_afternoon"
    assert intent["scene"] == "family"
    assert intent["location"]["max_distance_km"] == 8.0
    assert "kid_friendly" in intent["constraints"]["hard"]
    assert "low_calorie" in intent["constraints"]["soft"]
    assert "budget" in intent["missing_slots"]


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
    assert result["short_term_memory"]
