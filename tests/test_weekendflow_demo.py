from src.graph import get_graph
from src.nodes.intent_parser import constraints_from_intent, parse_intent
from src.nodes.memory_manager import apply_value_memory, load_memory
from src.nodes.payment_layer import _build_payable_items, payment_layer_node
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


def test_payment_layer_only_pays_itinerary_destinations() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "poi_id": "act_001",
                    "activity": "亲子陶艺体验馆",
                    "time": "14:00-15:30",
                },
                {
                    "poi_id": "rest_001",
                    "activity": "轻食餐厅",
                    "time": "17:00-18:00",
                },
            ]
        },
        "action_sequence": [
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "act_001",
                "time": "14:00",
            },
            {
                "step": 2,
                "action_type": "reserve_restaurant",
                "poi_id": "rest_001",
                "time": "17:00",
            },
            {
                "step": 3,
                "action_type": "order_addon_service",
                "time": "18:00",
            },
        ],
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "action": "order_activity_ticket",
                "name": "亲子陶艺体验馆",
                "data": {
                    "success": True,
                    "order_id": "act_001_14:00_TEST",
                    "time_slot": "14:00",
                    "quantity": 3,
                    "total_price": 594,
                    "payment_required": True,
                },
            },
            "reserve_restaurant_2": {
                "success": True,
                "action": "reserve_restaurant",
                "name": "轻食餐厅",
                "data": {
                    "success": True,
                    "order_id": "rest_001_17:00_TEST",
                    "payment_required": False,
                },
            },
            "order_addon_service_3": {
                "success": True,
                "action": "order_addon_service",
                "name": "庆祝蛋糕",
                "data": {
                    "success": True,
                    "order_id": "ADDON_cake_TEST",
                    "price": 88,
                    "payment_required": True,
                },
            },
        },
        "execution_log": [],
    }

    payable_items = _build_payable_items(state)

    assert len(payable_items) == 1
    assert payable_items[0]["action_type"] == "order_activity_ticket"
    assert payable_items[0]["name"] == "亲子陶艺体验馆"


def test_payment_layer_marks_payable_destination_success() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "poi_id": "act_001",
                    "activity": "亲子陶艺体验馆",
                    "time": "14:00-15:30",
                }
            ]
        },
        "action_sequence": [
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "act_001",
                "time": "14:00",
            }
        ],
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "action": "order_activity_ticket",
                "name": "亲子陶艺体验馆",
                "data": {
                    "success": True,
                    "order_id": "act_001_14:00_TESTPAY",
                    "time_slot": "14:00",
                    "quantity": 1,
                    "total_price": 198,
                    "payment_required": True,
                },
            }
        },
        "payment_ui_mode": "auto",
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "execution_log": [],
    }

    from src.tools.mock_apis import db

    db.orders["act_001_14:00_TESTPAY"] = {
        "poi_id": "act_001",
        "time_slot": "14:00",
        "quantity": 1,
        "status": "confirmed",
        "payment_status": "unpaid",
    }

    result = payment_layer_node(state)

    assert result["payment_status"] == "success"
    assert result["payment_results"]["payment_status"] == "paid"
    assert db.orders["act_001_14:00_TESTPAY"]["payment_status"] == "paid"
