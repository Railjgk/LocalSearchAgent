from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.tool_router import tool_router_node
import src.nodes.candidate_generator as candidate_generator_module


def test_rag_node_candidates_enable_mixed_domain_multinode_plan():
    state = {
        "user_input": "我们一家想在外滩附近住一晚，晚上吃蟹黄面，再找便利店买日用品，最后找停车场。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "住宿 晚餐 蟹黄面 便利店 停车",
            "people_count": 4,
            "budget": 1800,
            "max_distance_km": 25,
            "max_queue_time_min": 60,
            "origin_coordinates": "121.490,31.240",
        },
        "b_rag_node_candidates": {
            "lodging": [
                {
                    "poi_id": "rag_hotel_001",
                    "name": "外滩家庭套房酒店",
                    "category": "亲子酒店",
                    "coordinates": "121.492,31.242",
                    "price": 980,
                    "rating": 4.7,
                    "tags": ["家庭房", "有厨房", "近外滩"],
                    "evidence": ["家庭套房", "步行到外滩约12分钟"],
                    "available": True,
                }
            ],
            "restaurant": [
                {
                    "poi_id": "rag_restaurant_001",
                    "name": "沪上蟹黄面馆",
                    "category": "蟹黄面",
                    "restaurant_category": "本帮面馆",
                    "coordinates": "121.498,31.239",
                    "price": 96,
                    "rating": 4.6,
                    "queue_time_min": 15,
                    "duration_min": 70,
                    "parking_available": True,
                    "tags": ["蟹黄面", "正餐", "本帮菜"],
                    "evidence": ["招牌蟹黄面", "晚餐可堂食"],
                    "available_slots": [{"time": "18:30", "seats": 4}],
                    "available": True,
                },
                {
                    "poi_id": "rag_restaurant_002",
                    "name": "外滩轻食简餐",
                    "category": "简餐",
                    "restaurant_category": "轻食",
                    "coordinates": "121.496,31.241",
                    "price": 72,
                    "rating": 4.4,
                    "queue_time_min": 5,
                    "duration_min": 60,
                    "parking_available": True,
                    "tags": ["轻食", "简餐"],
                    "evidence": ["可快速用餐"],
                    "available_slots": [{"time": "18:00", "seats": 4}],
                    "available": True,
                },
            ],
            "convenience_store": [
                {
                    "poi_id": "rag_shop_001",
                    "name": "全时便利外滩店",
                    "category": "便利店",
                    "coordinates": "121.499,31.238",
                    "price": 40,
                    "rating": 4.2,
                    "duration_min": 20,
                    "tags": ["便利店", "日用品", "饮用水"],
                    "evidence": ["24小时营业", "可购买日用品"],
                    "available": True,
                }
            ],
            "parking": [
                {
                    "poi_id": "rag_parking_001",
                    "name": "外滩源停车场",
                    "category": "停车场",
                    "coordinates": "121.491,31.241",
                    "price": 35,
                    "rating": 4.1,
                    "duration_min": 15,
                    "tags": ["停车场", "近外滩"],
                    "evidence": ["夜间有余位"],
                    "available": True,
                }
            ],
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["b_itinerary_blueprint"]["template_mode"] == "multi_node"
    assert state["b_rag_candidate_coverage"]["unsupported_roles_covered"] is True
    assert state["candidates"]
    assert state["candidates"][0]["planner_mode"] == "multi_node_itinerary"
    assert state["candidates"][0]["execution_scope"] == "partial"
    assert any(node.get("supply_domain") == "hotel" for node in state["candidates"][0]["nodes"])

    state.update(constraint_filter_node(state))
    assert state["filtered_candidates"]

    state.update(plan_optimizer_node(state))
    selected_plan = state["selected_plan"]
    assert selected_plan["plan_shape"] == "multi_node"
    assert selected_plan["benchmark_ready"] is True
    assert selected_plan["execution_scope"] == "partial"
    assert selected_plan["execution_ready"] is False
    assert any(item.get("type") == "lodging" for item in selected_plan["timeline"])
    assert any(item.get("type") == "shopping" for item in selected_plan["timeline"])
    assert any(item.get("type") == "transport" for item in selected_plan["timeline"])

    state.update(tool_router_node(state))
    action_types = [item["action_type"] for item in state["action_sequence"]]
    assert "reserve_restaurant" in action_types
    non_executable_poi_ids = {
        item["poi_id"]
        for item in selected_plan["non_executable_nodes"]
        if item.get("poi_id")
    }
    assert not any(
        item.get("poi_id") in non_executable_poi_ids
        for item in state["action_sequence"]
    )


def test_rag_candidate_evidence_contract_from_handoff_is_consumed_by_b():
    state = {
        "user_input": "我想参加上海红翠斗芳菲漆器展，然后在附近吃个中饭，还想买点上海特产带回去。",
        "scene_type": "solo",
        "constraints": {
            "raw_text": "上海 红翠斗芳菲 漆器展 中饭 上海特产",
            "people_count": 1,
            "budget": 500,
            "max_distance_km": 20,
            "max_queue_time_min": 45,
            "origin_coordinates": "121.475,31.230",
        },
        "b_rag_candidate_evidence": {
            "version": "b_rag_candidate_evidence_v1",
            "query": "红翠斗芳菲漆器展 + 午餐 + 上海特产",
            "city": "上海",
            "blueprint_version": "b_itinerary_blueprint_v1",
            "node_evidence": [
                {
                    "node_id": "intent_01",
                    "role": "exhibition",
                    "label": "展览/博物馆",
                    "supply_domain": "activity",
                    "coverage_status": "covered",
                    "search_queries": ["红翠斗芳菲 宋元明漆器珍品展 上海"],
                    "candidates": [
                        {
                            "poi_id": "rag_activity_lacquer",
                            "amap_id": "amap_lacquer",
                            "merchant_id": "merchant_lacquer",
                            "name": "上海博物馆人民广场馆",
                            "city": "上海",
                            "district": "黄浦区",
                            "business_area": "人民广场",
                            "address": "人民大道201号",
                            "coordinates": "121.475,31.230",
                            "category": "博物馆展览",
                            "rating": 4.7,
                            "price": 0,
                            "business_hours": "09:00-17:00",
                            "retrieval_score": 0.92,
                            "evidence_text": "匹配漆器展和博物馆需求，位于人民广场。",
                            "field_sources": {
                                "identity": "observed_gaode",
                                "coordinates": "observed_gaode",
                                "price": "observed_gaode_or_mock",
                            },
                            "available": True,
                        }
                    ],
                },
                {
                    "node_id": "intent_02",
                    "role": "restaurant_lunch",
                    "label": "午餐",
                    "supply_domain": "restaurant",
                    "coverage_status": "covered",
                    "search_queries": ["人民广场 附近 中饭 简餐"],
                    "candidates": [
                        {
                            "poi_id": "rag_lunch_001",
                            "name": "人民广场本帮简餐",
                            "coordinates": "121.477,31.231",
                            "category": "本帮菜",
                            "restaurant_category": "本帮菜",
                            "price": 88,
                            "rating": 4.5,
                            "queue_time_min": 10,
                            "duration_min": 70,
                            "tags": ["午餐", "本帮菜", "正餐"],
                            "evidence_text": "人民广场附近可堂食午餐。",
                            "available_slots": [{"time": "12:30", "seats": 1}],
                            "available": True,
                        }
                    ],
                },
                {
                    "node_id": "intent_03",
                    "role": "souvenir_shopping",
                    "label": "特产/伴手礼",
                    "supply_domain": "shopping",
                    "coverage_status": "covered",
                    "search_queries": ["人民广场 上海特产 伴手礼"],
                    "candidates": [
                        {
                            "poi_id": "rag_souvenir_001",
                            "name": "上海特产伴手礼店",
                            "coordinates": "121.479,31.232",
                            "category": "伴手礼",
                            "price": 120,
                            "rating": 4.3,
                            "duration_min": 35,
                            "tags": ["上海特产", "伴手礼"],
                            "evidence_text": "可购买上海本地伴手礼。",
                            "available": True,
                        }
                    ],
                },
            ],
            "retrieval_trace": [{"step": "retrieve_evidence", "retriever": "hybrid"}],
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))

    assert state["b_rag_candidate_metadata"]["normalized_candidate_count"] >= 3
    assert "b_rag_candidate_evidence" in state["b_rag_candidate_metadata"]["supported_input_keys"]
    assert state["b_rag_candidate_coverage"]["covered_node_count"] >= 3
    assert state["candidates"]
    assert state["candidates"][0]["planner_mode"] == "multi_node_itinerary"
    assert {
        node.get("itinerary_role")
        for node in state["candidates"][0]["nodes"]
    } >= {"exhibition", "restaurant_lunch", "souvenir_shopping"}
    assert any(
        node.get("poi_id") == "rag_activity_lacquer"
        and node.get("type") == "activity"
        and node.get("itinerary_role") == "exhibition"
        for node in state["candidates"][0]["nodes"]
    )
    assert any(
        node.get("poi_id") == "rag_activity_lacquer"
        for node in state["candidates"][0]["nodes"]
    )


def test_rag_covered_multinode_uses_fast_path_without_full_supply_fetch(monkeypatch):
    def fail_fetch(**_kwargs):
        raise AssertionError("RAG-covered multi-node planning should not fetch full local supply")

    monkeypatch.setattr(candidate_generator_module, "fetch_activity_candidates", fail_fetch)
    monkeypatch.setattr(candidate_generator_module, "fetch_restaurant_candidates", fail_fetch)

    blueprint = {
        "version": "b_itinerary_blueprint_v1",
        "template_mode": "multi_node",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "node_count": 2,
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "family_activity",
                "label": "亲子活动",
                "supply_domain": "activity",
                "default_duration_min": 90,
                "sequence_index": 1,
            },
            {
                "node_id": "intent_02",
                "role": "restaurant_dinner",
                "label": "晚餐",
                "supply_domain": "restaurant",
                "default_duration_min": 80,
                "sequence_index": 2,
            },
        ],
        "time_skeleton": {
            "planning_days": 1,
            "planning_horizon": "half_day",
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "role": "family_activity",
                            "start_time": "14:00",
                            "end_time": "15:30",
                            "duration_min": 90,
                        },
                        {
                            "node_id": "intent_02",
                            "role": "restaurant_dinner",
                            "start_time": "17:30",
                            "end_time": "18:50",
                            "duration_min": 80,
                        },
                    ],
                }
            ],
        },
        "unsupported_roles": [],
        "requires_rag": True,
    }
    state = {
        "user_input": "下午带孩子玩，再吃晚饭。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "下午 亲子 晚饭",
            "city": "上海",
            "people_count": 3,
            "budget": 600,
            "origin_coordinates": "121.470,31.230",
            "b_itinerary_blueprint": blueprint,
        },
        "b_rag_candidate_evidence": {
            "version": "b_rag_candidate_evidence_v1",
            "node_evidence": [
                {
                    "node_id": "intent_01",
                    "role": "family_activity",
                    "supply_domain": "activity",
                    "coverage_status": "covered",
                    "candidates": [
                        {
                            "poi_id": "rag_act_001",
                            "name": "树屋亲子手作馆",
                            "type": "activity",
                            "category": "亲子手作",
                            "coordinates": "121.471,31.231",
                            "price": 120,
                            "rating": 4.7,
                            "duration_min": 90,
                            "tags": ["亲子", "儿童友好", "低强度"],
                            "available": True,
                        }
                    ],
                },
                {
                    "node_id": "intent_02",
                    "role": "restaurant_dinner",
                    "supply_domain": "restaurant",
                    "coverage_status": "covered",
                    "candidates": [
                        {
                            "poi_id": "rag_res_001",
                            "name": "禾间轻食餐厅",
                            "type": "restaurant",
                            "category": "轻食",
                            "restaurant_category": "轻食",
                            "coordinates": "121.474,31.232",
                            "price": 160,
                            "rating": 4.6,
                            "duration_min": 80,
                            "queue_time_min": 5,
                            "tags": ["晚餐", "轻食", "儿童椅"],
                            "available": True,
                        }
                    ],
                },
            ],
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    updates = candidate_generator_module.candidate_generator_node(state)

    assert updates["candidates"]
    assert updates["candidate_recall_diagnostics"]["fast_path"] == "rag_only_multinode"
    assert updates["candidates"][0]["planner_mode"] == "multi_node_itinerary"
