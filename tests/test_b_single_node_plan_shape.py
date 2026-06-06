from src.nodes.b_requirement_compiler import generate_b_requirement_contract
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node


def _run_b(state: dict) -> dict:
    state.update(candidate_generator_node(state))
    state.update(constraint_filter_node(state))
    state.update(plan_optimizer_node(state))
    return state


def test_restaurant_only_hotpot_request_does_not_force_activity():
    text = "今晚朋友聚餐想吃火锅，能坐下来聊天，不要烧烤。"
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "people_count": 4,
            "planning_preferences": {"food_type": ["火锅"]},
            "avoid": ["烧烤", "烤肉"],
            "budget": 900,
        },
        "scenario_activities": ["火锅", "聊天"],
        "user_profile": {},
        "execution_log": [],
    }

    _run_b(state)

    selected_plan = state["selected_plan"]
    assert selected_plan["planner_mode"] == "single_node"
    assert selected_plan["plan_shape"] == "restaurant_only"
    assert [item["type"] for item in selected_plan["timeline"]] == ["restaurant"]
    assert [item["action_type"] for item in selected_plan["action_hints"]] == ["reserve_restaurant"]


def test_cafe_only_request_keeps_only_cafe_node():
    text = "下午想一个人找个咖啡店坐一会儿，不吃正餐，安静一点。"
    state = {
        "user_input": text,
        "scene_type": "solo",
        "constraints": {
            "raw_text": text,
            "people_count": 1,
            "planning_preferences": {"food_type": ["咖啡", "甜品"]},
            "budget": 180,
        },
        "scenario_activities": ["咖啡", "安静"],
        "user_profile": {},
        "execution_log": [],
    }

    _run_b(state)

    selected_plan = state["selected_plan"]
    assert selected_plan["planner_mode"] == "single_node"
    assert selected_plan["plan_shape"] == "cafe_only"
    assert [item["type"] for item in selected_plan["timeline"]] == ["restaurant"]
    assert [item["action_type"] for item in selected_plan["action_hints"]] == ["reserve_restaurant"]


def test_restaurant_then_explicit_activity_keeps_pair_guardrail():
    text = "今晚和朋友吃火锅，5个人，预算600，吃完火锅去运动"
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "people_count": 5,
            "budget": 600,
            "planning_preferences": {
                "food_type": ["火锅"],
                "activity_type": ["运动"],
            },
        },
        "scenario_activities": [],
        "user_profile": {},
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["candidate_recall_diagnostics"]["single_node_shape"] is None
    assert state["candidates"]
    assert all(len(plan.get("nodes", []) or []) == 2 for plan in state["candidates"])


def test_negation_compiler_handles_do_not_recommend_hotpot():
    text = "今晚和朋友想吃烤肉，最好是炭火或者日式烧肉，别推荐火锅。"
    contract, _metadata = generate_b_requirement_contract(
        {"user_input": text, "constraints": {"raw_text": text}},
        constraints={"raw_text": text},
    )

    assert "火锅" in contract["forbidden_restaurant_groups"]


def test_negation_compiler_handles_do_not_arrange_bbq_hotpot():
    text = "今晚6个同事过生日，想先吃饭再去KTV，别安排烤肉火锅那种容易踩雷的。"
    contract, _metadata = generate_b_requirement_contract(
        {"user_input": text, "constraints": {"raw_text": text}},
        constraints={"raw_text": text},
    )

    assert {"烤肉", "火锅"} <= set(contract["forbidden_restaurant_groups"])


def test_long_duration_restaurant_request_keeps_pair_itinerary():
    text = "\u53ea\u8981\u5802\u98df\u8ba2\u5ea7\uff0c\u4e0d\u8981\u5916\u5e26\uff0c\u522b\u6392\u961f\u3002"
    state = {
        "user_input": text,
        "scene_type": "solo",
        "constraints": {
            "raw_text": text,
            "people_count": 1,
            "duration_range": [3, 6],
            "planning_preferences": {"restaurant_type": ["dine_in", "reservation_needed"]},
        },
        "scenario_activities": ["dine_in", "reservation_needed", "walk_in_ok"],
        "user_profile": {},
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["candidate_recall_diagnostics"]["single_node_shape"] is None


def test_default_a_duration_does_not_force_single_restaurant_into_pair():
    text = "\u4e2d\u5348\u60f3\u5403\u4e2a\u4fbf\u5b9c\u7684\u65e5\u5f0f\u732a\u6392\u5957\u9910\uff0c\u54ea\u5bb6\u5e97\u6700\u5b9e\u60e0\uff1f"
    state = {
        "user_input": text,
        "scene_type": "solo",
        "constraints": {
            "raw_text": text,
            "people_count": 1,
            "duration_range": [3, 6],
            "time_window": "unspecified",
            "confidence": {"time_window": 0.35},
            "planning_preferences": {
                "activity_type": ["\u8f7b\u91cf\u6d3b\u52a8"],
                "food_type": ["\u65e5\u5f0f\u732a\u6392", "\u5957\u9910"],
            },
        },
        "scenario_activities": ["\u8f7b\u91cf\u6d3b\u52a8", "\u653e\u677e", "\u9644\u8fd1"],
        "user_profile": {},
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["candidate_recall_diagnostics"]["single_node_shape"] == "restaurant_only"
    assert all(len(plan.get("nodes", []) or []) == 1 for plan in state["candidates"])


def test_business_area_park_name_does_not_force_restaurant_pair():
    text = (
        "\u6211\u5728\u9f99\u4e4b\u68a6\u9644\u8fd1\u5de5\u4f5c\uff0c"
        "\u4e2d\u5348\u60f3\u5403\u4e2a\u4fbf\u5b9c\u7684\u65e5\u5f0f\u732a\u6392\u5957\u9910\uff0c"
        "\u73af\u7403\u6e2f\u3001\u4e2d\u5c71\u516c\u56ed\u3001\u9759\u5b89\u5bfa\u8fd9\u4e09\u4e2a\u5546\u5708\u54ea\u5bb6\u5e97\u6700\u5b9e\u60e0\uff1f"
    )
    state = {
        "user_input": text,
        "scene_type": "low_budget",
        "constraints": {
            "raw_text": text,
            "people_count": 1,
            "duration_range": [3, 6],
            "time_window": "unspecified",
            "confidence": {"time_window": 0.35},
            "planning_preferences": {"activity_type": ["\u9884\u7b97\u654f\u611f"]},
        },
        "scenario_activities": ["\u9884\u7b97\u654f\u611f", "\u4f4e\u9884\u7b97", "\u9644\u8fd1"],
        "user_profile": {},
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["candidate_recall_diagnostics"]["single_node_shape"] == "restaurant_only"
    assert all(len(plan.get("nodes", []) or []) == 1 for plan in state["candidates"])


def test_broader_outing_with_meal_keeps_pair_itinerary():
    text = "\u4eca\u5929\u548c\u670b\u53cb\u60f3\u5728\u9644\u8fd1\u627e\u4e2a\u597d\u73a9\u53c8\u522b\u592a\u6298\u817e\u7684\u5730\u65b9\uff0c\u987a\u4fbf\u5403\u4e2a\u996d\u3002"
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "people_count": 2,
            "planning_preferences": {},
        },
        "scenario_activities": ["\u5ba4\u5185", "\u793e\u4ea4"],
        "user_profile": {},
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["candidate_recall_diagnostics"]["single_node_shape"] is None
