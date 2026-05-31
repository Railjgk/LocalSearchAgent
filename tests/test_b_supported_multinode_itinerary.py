from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.tool_router import tool_router_node


def test_supported_activity_restaurant_multinode_builds_executable_itinerary():
    state = {
        "user_input": "上午带孩子做亲子活动，中午吃轻食，下午看展，晚上再吃一顿清淡晚餐，路线别太远。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "上午 亲子 活动 中午 轻食 下午 看展 晚上 晚餐",
            "child_age": 5,
            "mom_diet": "low_calorie",
            "people_count": 3,
            "max_distance_km": 15,
            "max_queue_time_min": 45,
            "budget": 1200,
            "planning_preferences": {
                "activity_type": ["亲子", "看展"],
                "food_type": ["轻食", "清淡"],
            },
        },
        "user_profile": {"food_preference": ["light_food"]},
        "scenario_activities": ["亲子", "看展", "轻食"],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["b_itinerary_blueprint"]["template_mode"] == "multi_node"
    assert state["b_itinerary_blueprint"]["planning_horizon"] == "full_day"
    assert state["candidates"]
    assert state["candidates"][0]["planner_mode"] == "multi_node_itinerary"
    assert len(state["candidates"][0]["nodes"]) >= 4

    state.update(constraint_filter_node(state))
    assert state["filtered_candidates"]

    state.update(plan_optimizer_node(state))
    selected_plan = state["selected_plan"]
    assert selected_plan["planner_mode"] == "multi_node_itinerary"
    assert selected_plan["plan_shape"] == "multi_node"
    assert selected_plan["planning_days"] == 1
    assert len([item for item in selected_plan["timeline"] if item.get("poi_id")]) >= 4
    assert len(selected_plan["action_hints"]) >= 4

    state.update(tool_router_node(state))
    action_types = [item["action_type"] for item in state["action_sequence"]]
    assert action_types.count("order_activity_ticket") >= 2
    assert action_types.count("reserve_restaurant") >= 2


def test_supported_two_day_activity_restaurant_itinerary_keeps_day_split():
    state = {
        "user_input": "周末两天轻松安排：第一天亲子活动和晚餐，第二天上午看展，中午吃轻食。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "两天 第一天 亲子 活动 晚餐 第二天 上午 看展 中午 轻食",
            "child_age": 5,
            "mom_diet": "low_calorie",
            "people_count": 3,
            "max_distance_km": 30,
            "max_queue_time_min": 45,
            "budget": 1600,
            "planning_preferences": {
                "activity_type": ["亲子", "看展"],
                "food_type": ["轻食"],
            },
        },
        "user_profile": {"food_preference": ["light_food"]},
        "scenario_activities": ["亲子", "看展", "轻食"],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))
    assert state["b_itinerary_blueprint"]["planning_days"] == 2
    assert state["candidates"]

    state.update(constraint_filter_node(state))
    assert state["filtered_candidates"]

    state.update(plan_optimizer_node(state))
    selected_plan = state["selected_plan"]
    assert selected_plan["planner_mode"] == "multi_node_itinerary"
    assert selected_plan["planning_days"] == 2
    assert {item.get("day") for item in selected_plan["timeline"]} == {1, 2}
