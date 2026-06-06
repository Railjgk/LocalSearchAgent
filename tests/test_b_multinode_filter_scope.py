from src.nodes.constraint_filter import constraint_filter_node


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
