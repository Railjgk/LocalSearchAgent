from src.nodes.constraint_filter import (
    SCHEDULE_FEASIBILITY_REJECT_REASON,
    constraint_filter_node,
)


def _base_multinode_plan(nodes):
    return {
        "plan_id": "cand_multi_scope",
        "planner_mode": "multi_node_itinerary",
        "plan_shape": "multi_node",
        "nodes": nodes,
        "route": {"total_distance_km": 6.0, "total_travel_time_min": 30, "legs": []},
        "budget": {"total_price": 360},
        "availability": {"all_available": True, "max_queue_time_min": 10},
        "estimated_duration_min": 420,
        "planning_horizon": "full_day",
        "planning_days": 1,
    }


def test_low_calorie_memory_accepts_dinner_with_low_oil_options():
    """A spouse diet memory should not kill a local dinner that can be made low-oil."""

    plan = _base_multinode_plan(
        [
            {
                "poi_id": "act_1",
                "type": "activity",
                "name": "亲子手作馆",
                "itinerary_role": "family_activity",
                "tags": ["儿童友好", "低强度"],
            },
            {
                "poi_id": "res_lunch",
                "type": "restaurant",
                "name": "轻食日料",
                "itinerary_role": "restaurant_lunch",
                "tags": ["轻食", "低卡"],
            },
            {
                "poi_id": "res_dinner",
                "type": "restaurant",
                "name": "上厨本帮菜",
                "itinerary_role": "restaurant_dinner",
                "tags": ["本帮家常菜", "家庭友好"],
                "health_tags": ["可少油"],
                "menu_health_options": ["少油清蒸菜", "时令蔬菜"],
            },
        ]
    )
    state = {
        "constraints": {
            "raw_text": "周末一整天，中午吃清淡一点，晚上吃本帮菜",
            "city": "上海",
        },
        "user_profile": {
            "companion_profile": {
                "wife": {"state": "dieting", "needs": ["low_calorie", "light_food"]}
            }
        },
        "candidates": [plan],
    }

    result = constraint_filter_node(state)

    assert len(result["filtered_candidates"]) == 1


def test_schedule_drift_available_pois_gets_schedule_specific_rejection():
    plan = _base_multinode_plan(
        [
            {
                "poi_id": "act_citywalk",
                "type": "activity",
                "name": "衡复城市漫步",
                "itinerary_role": "citywalk_market",
                "available": True,
            },
            {
                "poi_id": "res_dinner",
                "type": "restaurant",
                "name": "清淡素食馆",
                "itinerary_role": "restaurant_specific",
                "available": True,
            },
        ]
    )
    plan["availability"] = {
        "all_available": False,
        "max_queue_time_min": 10,
        "detail": {
            "unavailable_poi_ids": [],
            "time_window_feasible": False,
            "schedule_feasibility": {
                "time_window_feasible": False,
                "drifted_nodes": [
                    {
                        "node_id": "intent_01",
                        "label": "城市漫步/市集",
                        "role": "citywalk_market",
                        "slot_start": "10:30",
                        "slot_end": "12:30",
                        "scheduled_start": "14:00",
                        "scheduled_end": "16:00",
                        "reason": "slot_alignment_drift",
                    }
                ],
            },
        },
    }
    state = {
        "constraints": {"raw_text": "明天朋友 citywalk 拍照 晚餐 素食 不辣"},
        "user_profile": {},
        "candidates": [plan],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["cand_multi_scope"] == SCHEDULE_FEASIBILITY_REJECT_REASON
    assert "活动或餐厅当前不可用" not in result["filter_reasons"]["_summary_detail"]["reason_counts"]


def test_pet_contract_stays_primary_when_schedule_drift_also_exists():
    plan = _base_multinode_plan(
        [
            {
                "poi_id": "act_ritual",
                "type": "activity",
                "name": "安静手作体验",
                "itinerary_role": "cultural_photo",
                "available": True,
            },
            {
                "poi_id": "res_dinner",
                "type": "restaurant",
                "name": "纪念日晚餐",
                "itinerary_role": "restaurant_dinner",
                "available": True,
            },
        ]
    )
    plan["availability"] = {
        "all_available": False,
        "max_queue_time_min": 10,
        "detail": {
            "unavailable_poi_ids": [],
            "time_window_feasible": False,
        },
    }
    state = {
        "constraints": {"raw_text": "带狗 纪念日 两天一夜 停车"},
        "user_profile": {},
        "b_requirement_contract": {"hard_requirements": ["pet_friendly"]},
        "candidates": [plan],
    }

    result = constraint_filter_node(state)

    assert result["filter_reasons"]["cand_multi_scope"] == "缺少活动和餐厅均宠物友好的证据"
