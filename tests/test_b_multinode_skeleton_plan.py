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
