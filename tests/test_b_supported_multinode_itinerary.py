from src.nodes.candidate_generator import (
    _build_multinode_schedule,
    _explicit_restaurant_requirements,
    _matches_strict_node_role,
    _rank_pool_for_node_intent,
    candidate_generator_node,
)
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
    if state["filtered_candidates"]:
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
    else:
        assert "活动或餐厅当前不可用" in state["filter_reasons"].get("_summary", "")


def test_restaurant_specific_keeps_explicit_cuisine_as_hard_requirement():
    constraints = {
        "planning_preferences": {},
        "b_itinerary_blueprint": {
            "node_intents": [
                {
                    "node_id": "intent_01",
                    "role": "restaurant_specific",
                    "supply_domain": "restaurant",
                    "search_terms": ["\u706b\u9505"],
                }
            ]
        },
    }

    required = _explicit_restaurant_requirements(constraints)

    assert "\u706b\u9505" in required or "hotpot" in required


def test_restaurant_specific_ranking_filters_non_requested_cuisine():
    intent = {
        "node_id": "intent_01",
        "role": "restaurant_specific",
        "supply_domain": "restaurant",
        "search_terms": ["\u706b\u9505"],
    }
    restaurants = [
        {
            "poi_id": "res_local_food",
            "name": "\u5c0f\u7ecd\u5174",
            "type": "restaurant",
            "supply_domain": "restaurant",
            "restaurant_category": "\u672c\u5e2e\u83dc",
            "tags": ["\u672c\u5e2e\u83dc", "\u8001\u5b57\u53f7"],
            "rating": 4.9,
            "available": True,
        },
        {
            "poi_id": "res_hotpot",
            "name": "\u9759\u5b89\u725b\u8089\u706b\u9505",
            "type": "restaurant",
            "supply_domain": "restaurant",
            "restaurant_category": "\u706b\u9505",
            "tags": ["\u706b\u9505", "hotpot"],
            "rating": 4.2,
            "available": True,
        },
    ]

    ranked = _rank_pool_for_node_intent(intent, [], restaurants)

    assert [item["poi_id"] for item in ranked] == ["res_hotpot"]


def test_guidance_only_nodes_do_not_block_c_execution_contract():
    candidate = {
        "plan_id": "cand_guidance_nodes",
        "scene_type": "solo",
        "planner_mode": "multi_node_itinerary",
        "plan_shape": "multi_node",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "benchmark_ready": True,
        "execution_scope": "full",
        "nodes": [
            {
                "poi_id": "act_1",
                "merchant_id": "m_act_1",
                "product_ids": ["prod_act_1"],
                "deal_ids": ["deal_act_1"],
                "type": "activity",
                "supply_domain": "activity",
                "itinerary_role": "exhibition",
                "name": "\u4e0a\u6d77\u5c55\u89c8",
                "available_slots": [{"time": "14:00", "inventory_left": 3}],
                "available": True,
                "price": 100,
                "rating": 4.6,
                "queue_time_min": 5,
            },
            {
                "poi_id": "res_1",
                "merchant_id": "m_res_1",
                "product_ids": ["prod_res_1"],
                "deal_ids": ["deal_res_1"],
                "type": "restaurant",
                "supply_domain": "restaurant",
                "itinerary_role": "restaurant_specific",
                "name": "\u9644\u8fd1\u70e7\u70e4",
                "available_slots": [{"time": "16:00", "seats": 1}],
                "reservation_slots": [{"time": "16:00", "seats": 1}],
                "available": True,
                "dine_in_available": True,
                "price": 120,
                "rating": 4.5,
                "queue_time_min": 8,
            },
            {
                "poi_id": "shop_1",
                "type": "shopping",
                "supply_domain": "shopping",
                "itinerary_role": "souvenir_shopping",
                "name": "\u987a\u8def\u4f34\u624b\u793c",
                "available": True,
                "price": 50,
                "rating": 4.3,
                "queue_time_min": 3,
            },
            {
                "poi_id": "parking_1",
                "type": "transport_service",
                "supply_domain": "transport_service",
                "itinerary_role": "parking",
                "name": "\u9644\u8fd1\u505c\u8f66\u573a",
                "available": True,
                "price": 0,
                "rating": 4.0,
                "queue_time_min": 0,
            },
        ],
        "timeline": [
            {"time": "14:00-15:30", "type": "activity", "poi_id": "act_1", "name": "\u4e0a\u6d77\u5c55\u89c8"},
            {"time": "16:00-17:00", "type": "restaurant", "poi_id": "res_1", "name": "\u9644\u8fd1\u70e7\u70e4"},
            {"time": "17:10-17:35", "type": "shopping", "poi_id": "shop_1", "name": "\u987a\u8def\u4f34\u624b\u793c"},
            {"time": "17:40-17:50", "type": "transport", "poi_id": "parking_1", "name": "\u9644\u8fd1\u505c\u8f66\u573a"},
        ],
        "route": {"total_distance_km": 4.2, "total_travel_time_min": 28},
        "budget": {"total_price": 270},
        "availability": {"all_available": True, "max_queue_time_min": 8},
        "estimated_duration_min": 230,
    }
    state = {
        "scene_type": "solo",
        "constraints": {
            "raw_text": "\u770b\u5c55\u5403\u70e7\u70e4\u518d\u4e70\u4f34\u624b\u793c\uff0c\u770b\u4e00\u4e0b\u505c\u8f66",
            "people_count": 1,
            "budget": 500,
            "max_distance_km": 10,
            "max_queue_time_min": 30,
        },
        "candidates": [candidate],
        "filtered_candidates": [candidate],
        "execution_log": [],
    }

    result = plan_optimizer_node(state)
    selected = result["selected_plan"]

    assert selected["execution_scope"] == "full"
    assert selected["execution_ready"] is True
    assert selected.get("non_executable_nodes") == []
    assert {
        node.get("role") for node in selected.get("guidance_only_nodes", [])
    } >= {"souvenir_shopping", "parking"}
    assert [hint["action_type"] for hint in selected["action_hints"]] == [
        "order_activity_ticket",
        "reserve_restaurant",
    ]


def test_multinode_omits_deal_when_action_time_is_not_valid_for_deal():
    candidate = {
        "plan_id": "cand_bad_activity_time",
        "scene_type": "friends",
        "planner_mode": "multi_node_itinerary",
        "plan_shape": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "benchmark_ready": True,
        "nodes": [
            {
                "poi_id": "act_museum",
                "merchant_id": "m_act_museum",
                "product_ids": ["prod_museum"],
                "deal_ids": ["deal_museum"],
                "type": "activity",
                "itinerary_role": "cultural_photo",
                "name": "Museum visit",
                "available_slots": [{"time": "14:00", "inventory_left": 20}],
                "products": [
                    {
                        "product_id": "prod_museum",
                        "poi_id": "act_museum",
                        "merchant_id": "m_act_museum",
                        "product_type": "activity_ticket",
                        "requires_reservation": False,
                    }
                ],
                "deals": [
                    {
                        "deal_id": "deal_museum",
                        "product_id": "prod_museum",
                        "poi_id": "act_museum",
                        "valid_time": ["14:00"],
                        "requires_reservation": False,
                    }
                ],
                "available": True,
                "price": 60,
                "rating": 4.6,
                "queue_time_min": 5,
            },
            {
                "poi_id": "res_light",
                "merchant_id": "m_res_light",
                "product_ids": ["prod_res_light"],
                "deal_ids": ["deal_res_light"],
                "type": "restaurant",
                "supply_domain": "restaurant",
                "itinerary_role": "restaurant_lunch",
                "name": "Light lunch",
                "available_slots": [{"time": "12:30", "seats": 4}],
                "reservation_slots": [{"time": "12:30", "seats": 4}],
                "available": True,
                "dine_in_available": True,
                "price": 120,
                "rating": 4.5,
                "queue_time_min": 5,
            },
        ],
        "timeline": [
            {"time": "10:00-12:00", "type": "activity", "poi_id": "act_museum", "name": "Museum visit"},
            {"time": "12:30-13:30", "type": "restaurant", "poi_id": "res_light", "name": "Light lunch"},
        ],
        "route": {"total_distance_km": 3.0, "total_travel_time_min": 20},
        "budget": {"total_price": 180},
        "availability": {"all_available": True, "max_queue_time_min": 5},
        "estimated_duration_min": 210,
    }
    state = {
        "scene_type": "friends",
        "constraints": {
            "raw_text": "museum then lunch",
            "people_count": 2,
            "budget": 500,
            "max_distance_km": 10,
            "max_queue_time_min": 30,
        },
        "candidates": [candidate],
        "filtered_candidates": [candidate],
        "execution_log": [],
    }

    result = plan_optimizer_node(state)
    selected = result["selected_plan"]
    failed_checks = {
        check["name"]
        for check in selected["execution_contract"]["checks"]
        if check["status"] == "fail"
    }

    assert selected["action_hints"][0]["product_id"] == "prod_museum"
    assert selected["action_hints"][0]["deal_id"] is None
    assert "node_1_time_in_deal" not in failed_checks
    assert selected["execution_ready"] is True
    assert selected["nodes"]


def test_multinode_omits_loose_deal_id_without_deal_record():
    candidate = {
        "plan_id": "cand_loose_deal",
        "scene_type": "friends",
        "planner_mode": "multi_node_itinerary",
        "plan_shape": "multi_node",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "benchmark_ready": True,
        "nodes": [
            {
                "poi_id": "act_photo",
                "merchant_id": "m_act_photo",
                "product_ids": ["prod_photo"],
                "deal_ids": ["deal_photo"],
                "type": "activity",
                "itinerary_role": "cultural_photo",
                "name": "Photo studio",
                "available_slots": [{"time": "15:00", "inventory_left": 8}],
                "products": [
                    {
                        "product_id": "prod_photo",
                        "poi_id": "act_photo",
                        "merchant_id": "m_act_photo",
                        "product_type": "activity_ticket",
                    }
                ],
                "available": True,
                "price": 120,
                "rating": 4.6,
                "queue_time_min": 5,
            }
        ],
        "timeline": [
            {
                "time": "15:00-16:30",
                "type": "activity",
                "poi_id": "act_photo",
                "name": "Photo studio",
            }
        ],
        "route": {"total_distance_km": 2.0, "total_travel_time_min": 10},
        "budget": {"total_price": 120},
        "availability": {"all_available": True, "max_queue_time_min": 5},
        "estimated_duration_min": 90,
    }
    state = {
        "scene_type": "friends",
        "constraints": {
            "raw_text": "photo activity",
            "people_count": 2,
            "budget": 500,
            "max_distance_km": 10,
            "max_queue_time_min": 30,
        },
        "candidates": [candidate],
        "filtered_candidates": [candidate],
        "execution_log": [],
    }

    selected = plan_optimizer_node(state)["selected_plan"]

    assert selected["nodes"][0]["poi_id"] == "act_photo"
    assert selected["action_hints"][0]["product_id"] == "prod_photo"
    assert selected["action_hints"][0]["deal_id"] is None
    assert selected["execution_ready"] is True


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
    if state["filtered_candidates"]:
        state.update(plan_optimizer_node(state))
        selected_plan = state["selected_plan"]
        assert selected_plan["planner_mode"] == "multi_node_itinerary"
        assert selected_plan["planning_days"] == 2
        assert {item.get("day") for item in selected_plan["timeline"]} == {1, 2}
    else:
        assert "活动或餐厅当前不可用" in state["filter_reasons"].get("_summary", "")


def test_multinode_schedule_uses_blueprint_slot_start_for_breakfast():
    breakfast = {
        "poi_id": "res_breakfast",
        "name": "\u65e9\u9910\u5e97",
        "type": "restaurant",
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "restaurant_breakfast",
            "label": "\u65e9\u9910",
            "default_duration_min": 45,
        },
        "itinerary_role": "restaurant_breakfast",
    }
    shop = {
        "poi_id": "shop_1",
        "name": "\u4fbf\u5229\u5e97",
        "type": "shopping",
        "_itinerary_intent": {
            "node_id": "intent_02",
            "role": "convenience_store",
            "label": "\u4fbf\u5229\u5e97",
            "default_duration_min": 20,
        },
        "itinerary_role": "convenience_store",
    }
    blueprint = {
        "planning_horizon": "half_day",
        "time_skeleton": {
            "days": [
                {
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "08:30",
                            "duration_min": 45,
                        },
                        {
                            "node_id": "intent_02",
                            "day": 1,
                            "start_time": "09:35",
                            "duration_min": 20,
                        },
                    ]
                }
            ]
        },
    }

    timeline, _ = _build_multinode_schedule([breakfast, shop], blueprint, {})

    assert timeline[0]["time"].startswith("08:30")
    assert timeline[1]["time"].startswith("09:")


def test_strict_multinode_role_checks_cultural_photo_identity():
    hanfu = {
        "poi_id": "act_hanfu",
        "name": "\u8001\u4e0a\u6d77\u65d7\u888d\u6c49\u670d",
        "type": "activity",
        "category": "\u6c49\u670d\u62cd\u7167",
    }
    spa = {
        "poi_id": "act_spa",
        "name": "\u6cf0\u5b81\u517b\u751f",
        "type": "activity",
        "category": "SPA\u6309\u6469",
    }

    assert _matches_strict_node_role(hanfu, "cultural_photo") is True
    assert _matches_strict_node_role(spa, "cultural_photo") is False


def test_cultural_photo_accepts_museum_culture_evidence():
    museum = {
        "poi_id": "act_museum",
        "name": "\u4e0a\u6d77\u5e02\u5386\u53f2\u535a\u7269\u9986",
        "type": "activity",
        "category": "\u535a\u7269\u9986",
        "tags": ["museum", "local_culture", "indoor"],
    }

    assert _matches_strict_node_role(museum, "cultural_photo") is True


def test_optimizer_prefers_lower_friction_full_day_food_sequence():
    def node(name, role, node_type, tags, price=80):
        return {
            "poi_id": f"{role}_{name[:2]}",
            "name": name,
            "type": node_type,
            "itinerary_role": role,
            "tags": tags,
            "rating": 4.6,
            "price": price,
            "available": True,
        }

    citywalk = node(
        "青岛小鱼山文化名人街区",
        "citywalk_market",
        "activity",
        ["城市漫步", "历史文化"],
        50,
    )
    park = node(
        "牡丹园",
        "park_scenic_walk",
        "activity",
        ["公园", "低强度"],
        0,
    )
    heavy_lunch = node(
        "王姐烧烤-海鲜家常菜",
        "restaurant_lunch",
        "restaurant",
        ["海鲜", "烧烤", "高热量", "油烟味", "排队久"],
        100,
    )
    light_lunch = node(
        "双合园-海鲜水饺青岛菜",
        "restaurant_lunch",
        "restaurant",
        ["海鲜", "青岛菜", "海鲜水饺", "清淡"],
        80,
    )
    dinner = node(
        "海鲜大排档",
        "restaurant_dinner",
        "restaurant",
        ["海鲜", "大排档", "烧烤", "油烟味"],
        120,
    )

    def plan(plan_id, lunch):
        nodes = [citywalk, lunch, park, dinner]
        return {
            "plan_id": plan_id,
            "planner_mode": "multi_node_itinerary",
            "plan_shape": "multi_node",
            "execution_scope": "full",
            "planning_horizon": "full_day",
            "planning_days": 1,
            "nodes": nodes,
            "timeline": [],
            "route": {
                "total_distance_km": 6,
                "total_travel_time_min": 45,
                "legs": [{"distance_km": 2}, {"distance_km": 2}, {"distance_km": 2}],
            },
            "budget": {"total_price": sum(item.get("price", 0) for item in nodes)},
            "availability": {"all_available": True, "max_queue_time_min": 10},
            "estimated_duration_min": 480,
            "tags": [tag for item in nodes for tag in item.get("tags", [])],
        }

    result = plan_optimizer_node(
        {
            "scene_type": "solo",
            "constraints": {
                "raw_text": "青岛轻松玩一天，想吃海鲜，别太累",
                "time_window": "full_day",
                "duration_range": [6, 10],
                "budget": 800,
                "max_distance_km": 15,
                "max_queue_time_min": 30,
            },
            "filtered_candidates": [
                plan("heavy_lunch", heavy_lunch),
                plan("lighter_lunch", light_lunch),
            ],
            "execution_log": [],
        }
    )

    selected = result["selected_plan"]

    assert selected["plan_id"] == "lighter_lunch"
    assert "轻松需求下存在排队、拥挤或油烟风险" in selected["risk_factors"]


def test_optimizer_returns_time_adjustment_fallback_for_late_night_closed_nodes():
    candidate = {
        "plan_id": "cand_late_tea",
        "planner_mode": "multi_node_itinerary",
        "plan_shape": "multi_node",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "nodes": [
            {
                "poi_id": "res_tea",
                "name": "\u8336\u7a7a\u95f4",
                "type": "restaurant",
                "itinerary_role": "tea_house",
                "supply_domain": "restaurant",
            },
            {
                "poi_id": "res_western",
                "name": "\u897f\u9910\u5385",
                "type": "restaurant",
                "itinerary_role": "restaurant_specific",
                "supply_domain": "restaurant",
            },
            {
                "poi_id": "res_cafe",
                "name": "\u5496\u5561\u9986",
                "type": "restaurant",
                "itinerary_role": "cafe",
                "supply_domain": "restaurant",
            },
        ],
        "timeline": [
            {"time": "22:00-23:00", "type": "restaurant", "activity": "\u8336\u7a7a\u95f4", "poi_id": "res_tea"},
            {"time": "23:20-00:30", "type": "restaurant", "activity": "\u897f\u9910\u5385", "poi_id": "res_western"},
            {"time": "00:45-01:30", "type": "restaurant", "activity": "\u5496\u5561\u9986", "poi_id": "res_cafe"},
        ],
        "route": {"total_distance_km": 2.8},
        "budget": {"total_price": 360},
        "availability": {"all_available": True},
        "estimated_duration_min": 210,
    }
    state = {
        "scene_type": "solo",
        "constraints": {"raw_text": "\u665a\u4e0a10\u70b9\u5916\u6ee9\u559d\u8336\u897f\u9910\u5496\u5561", "start_time": "22:00"},
        "candidates": [candidate],
        "filtered_candidates": [],
        "filter_reasons": {
            "cand_late_tea": "\u8425\u4e1a\u65f6\u95f4\u4e0d\u6ee1\u8db3\u6df1\u591c/\u591c\u5bb5\u9700\u6c42",
            "_summary": "\u5168\u90e8\u5019\u9009\u56e0\u6df1\u591c\u8425\u4e1a\u65f6\u95f4\u88ab\u8fc7\u6ee4",
        },
        "b_itinerary_blueprint": {
            "template_mode": "multi_node",
            "planning_horizon": "half_day",
            "planning_days": 1,
            "node_intents": [
                {"role": "tea_house"},
                {"role": "restaurant_specific"},
                {"role": "cafe"},
            ],
        },
        "execution_log": [],
    }

    result = plan_optimizer_node(state)
    selected = result["selected_plan"]

    assert selected["plan_status"] == "time_adjustment_required"
    assert selected["execution_ready"] is False
    assert selected["time_adjustment"]["requested_time_infeasible"] is True
    assert len([item for item in selected["timeline"] if item.get("poi_id")]) == 3
    assert selected["action_hints"] == []
