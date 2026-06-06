from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.explainability import explainability_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.tool_router import tool_router_node


def test_multinode_request_returns_non_executable_skeleton_without_legacy_pair():
    state = {
        "user_input": "我们一家四口想在外滩附近找个有厨房的住处，晚上吃蟹黄面，再找便利店买日用品，最后找停车场。",
        "scene_type": "family",
        "constraints": {"raw_text": "住宿 晚餐 便利店 停车"},
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))
    assert state["candidates"]
    assert state["b_itinerary_blueprint"]["template_mode"] == "multi_node"
    assert state["b_itinerary_blueprint"]["planning_days"] == 2
    assert state["candidate_generation_issues"][0]["type"] == "multi_node_blueprint_not_yet_planned"

    state.update(constraint_filter_node(state))
    state.update(plan_optimizer_node(state))

    selected_plan = state["selected_plan"]
    assert selected_plan["plan_status"] == "partial_executable"
    assert selected_plan["execution_ready"] is False
    assert selected_plan["planning_days"] == 2
    assert selected_plan["timeline"]
    assert any(item.get("poi_id") for item in selected_plan["timeline"])
    assert selected_plan["partial_missing_roles"]

    state.update(explainability_node(state))
    assert state["explanation_text"]

    state.update(tool_router_node(state))
    assert state["action_sequence"] == []


def test_multinode_skeleton_timeline_is_chronological_within_day():
    state = {
        "scene_type": "friends",
        "constraints": {},
        "filtered_candidates": [],
        "candidates": [],
        "filter_reasons": {},
        "execution_log": [],
        "b_itinerary_blueprint": {
            "template_mode": "multi_node",
            "planning_horizon": "full_day",
            "planning_days": 1,
            "time_skeleton": {
                "days": [
                    {
                        "day": 1,
                        "slots": [
                            {
                                "node_id": "intent_03",
                                "label": "下午补充活动",
                                "role": "cultural_photo",
                                "supply_domain": "activity",
                                "start_time": "16:30",
                                "end_time": "18:00",
                                "duration_min": 90,
                            },
                            {
                                "node_id": "intent_04",
                                "label": "傍晚餐饮/休息",
                                "role": "cafe",
                                "supply_domain": "restaurant",
                                "start_time": "15:30",
                                "end_time": "16:30",
                                "duration_min": 60,
                            },
                        ],
                    }
                ]
            },
        },
    }

    result = plan_optimizer_node(state)

    assert [item["time"] for item in result["selected_plan"]["timeline"]] == [
        "15:30-16:30",
        "16:30-18:00",
    ]
