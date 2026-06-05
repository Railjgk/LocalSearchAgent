from src.nodes.share_generator import share_generator_node
from src.nodes.tool_router import tool_router_node


def test_family_tool_router_does_not_add_unplanned_addon_service():
    state = {
        "scene_type": "family",
        "constraints": {"people_count": 2},
        "selected_plan": {
            "timeline": [
                {
                    "time": "15:30-17:00",
                    "activity": "树屋亲子陶艺体验馆",
                    "poi_id": "act_family_ceramic",
                },
                {
                    "time": "17:30-19:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                },
            ],
            "action_hints": [
                {
                    "action_type": "order_activity_ticket",
                    "poi_id": "act_family_ceramic",
                    "time": "15:30",
                    "quantity": 2,
                },
                {
                    "action_type": "reserve_restaurant",
                    "poi_id": "res_light_japanese",
                    "time": "17:30",
                    "people": 2,
                },
            ],
        },
        "execution_log": [],
    }

    routed = tool_router_node(state)

    assert [item["action_type"] for item in routed["action_sequence"]] == [
        "order_activity_ticket",
        "reserve_restaurant",
    ]
    assert all(
        item["action_type"] != "order_addon_service"
        for item in routed["action_sequence"]
    )


def test_tool_router_skips_full_plan_with_failed_execution_contract():
    state = {
        "scene_type": "friends",
        "constraints": {"people_count": 4},
        "selected_plan": {
            "execution_ready": False,
            "execution_scope": "full",
            "execution_contract": {
                "ready": False,
                "blocking_reasons": [
                    "node_3 action time must be valid for selected deal"
                ],
            },
            "timeline": [
                {
                    "time": "14:00-16:00",
                    "activity": "上海市历史博物馆",
                    "poi_id": "gaode_act_B0FFI2885X",
                },
                {
                    "time": "17:30-18:50",
                    "activity": "鮨士道寿司轻食(日月光中心店)",
                    "poi_id": "gaode_res_B0J1GD89GK",
                },
                {
                    "time": "19:20-20:50",
                    "activity": "上海田子坊",
                    "poi_id": "gaode_act_B00155HO6Y",
                },
            ],
            "action_hints": [
                {
                    "action_type": "order_activity_ticket",
                    "poi_id": "gaode_act_B0FFI2885X",
                    "time": "14:00",
                    "quantity": 4,
                },
                {
                    "action_type": "reserve_restaurant",
                    "poi_id": "gaode_res_B0J1GD89GK",
                    "time": "17:30",
                    "people": 4,
                },
                {
                    "action_type": "order_activity_ticket",
                    "poi_id": "gaode_act_B00155HO6Y",
                    "time": "19:20",
                    "quantity": 4,
                },
            ],
        },
        "execution_log": [],
    }

    routed = tool_router_node(state)

    assert routed["action_sequence"] == []
    assert "execution contract is not ready" in routed["execution_log"][-1]
    assert "node_3 action time" in routed["execution_log"][-1]


def test_tool_router_emits_supported_actions_for_partial_plan_with_missing_roles():
    state = {
        "scene_type": "friends",
        "constraints": {"people_count": 9},
        "selected_plan": {
            "plan_status": "partial_executable",
            "execution_ready": False,
            "execution_scope": "partial",
            "partial_missing_roles": ["souvenir_shopping", "flower_shop"],
            "execution_contract": {"ready": True, "blocking_reasons": []},
            "timeline": [
                {
                    "time": "18:30-20:30",
                    "activity": "邻里桌游咖啡馆",
                    "poi_id": "act_board_game_cafe",
                },
                {
                    "time": "21:00-22:00",
                    "activity": "青禾沙拉碗",
                    "poi_id": "res_salad_bowl",
                },
            ],
            "action_hints": [
                {
                    "action_type": "order_activity_ticket",
                    "poi_id": "act_board_game_cafe",
                    "time": "18:30",
                    "quantity": 9,
                },
                {
                    "action_type": "reserve_restaurant",
                    "poi_id": "res_salad_bowl",
                    "time": "21:00",
                    "people": 9,
                },
            ],
        },
        "execution_log": [],
    }

    routed = tool_router_node(state)

    assert [item["action_type"] for item in routed["action_sequence"]] == [
        "order_activity_ticket",
        "reserve_restaurant",
    ]
    assert routed["action_sequence"][0]["name"] == "邻里桌游咖啡馆"
    assert routed["action_sequence"][1]["name"] == "青禾沙拉碗"
    assert any(
        "emits supported actions for a partial B plan" in entry
        for entry in routed["execution_log"]
    )


def test_tool_router_infers_actions_for_concrete_multinode_timeline_without_types():
    state = {
        "scene_type": "friends",
        "constraints": {"people_count": 4},
        "selected_plan": {
            "timeline": [
                {
                    "time": "14:00-16:00",
                    "activity": "上海市历史博物馆",
                    "poi_id": "gaode_act_B0FFI2885X",
                    "notes": ["文化体验/拍照", "multi_node_itinerary"],
                },
                {
                    "time": "17:30-18:50",
                    "activity": "鮨士道寿司轻食(日月光中心店)",
                    "poi_id": "gaode_res_B0J1GD89GK",
                    "notes": ["晚餐", "multi_node_itinerary"],
                },
                {
                    "time": "19:20-20:50",
                    "activity": "上海田子坊",
                    "poi_id": "gaode_act_B00155HO6Y",
                    "notes": ["下午补充活动", "multi_node_itinerary"],
                },
            ],
        },
        "execution_log": [],
    }

    routed = tool_router_node(state)

    assert [item["action_type"] for item in routed["action_sequence"]] == [
        "order_activity_ticket",
        "reserve_restaurant",
        "order_activity_ticket",
    ]
    assert [item["time"] for item in routed["action_sequence"]] == [
        "14:00",
        "17:30",
        "19:20",
    ]
    assert routed["action_sequence"][1]["people"] == 4
    assert routed["action_sequence"][0]["quantity"] == 4


def test_tool_router_keeps_skeleton_timeline_without_pois_non_executable():
    state = {
        "scene_type": "family",
        "constraints": {"people_count": 3},
        "selected_plan": {
            "timeline": [
                {
                    "time": "10:00-12:00",
                    "activity": "亲子活动",
                    "poi_id": None,
                },
                {
                    "time": "18:00-19:20",
                    "activity": "晚餐",
                    "poi_id": None,
                },
            ],
        },
        "execution_log": [],
    }

    routed = tool_router_node(state)

    assert routed["action_sequence"] == []


def test_family_success_share_does_not_promise_unplanned_cake():
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "15:30-17:00",
                    "activity": "树屋亲子陶艺体验馆",
                    "poi_id": "act_family_ceramic",
                },
                {
                    "time": "17:30-19:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                },
            ]
        },
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "name": "树屋亲子陶艺体验馆",
                "data": {"action": "order_activity_ticket"},
            },
            "reserve_restaurant_2": {
                "success": True,
                "name": "禾间轻食日料",
                "data": {"action": "reserve_restaurant"},
            },
        },
        "execution_status": "success",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    assert "蛋糕" not in shared["final_share_message"]


def test_solo_no_alcohol_success_share_uses_neutral_arrival_copy():
    state = {
        "scene_type": "solo",
        "constraints": {
            "people_count": 1,
            "raw_text": "补牙后医生说两小时内别喝酒，想找个安静清淡晚饭。",
        },
        "selected_plan": {
            "timeline": [
                {
                    "time": "18:30-19:30",
                    "activity": "轻盈有机简餐",
                    "poi_id": "res_healthy_canteen",
                }
            ]
        },
        "tool_results": {
            "reserve_restaurant_1": {
                "success": True,
                "name": "轻盈有机简餐",
                "data": {"action": "reserve_restaurant"},
            }
        },
        "execution_status": "success",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "🍻" not in message
    assert "大家直接去" not in message
    assert "你按确认时间过去即可" in message


def test_failed_booking_share_includes_repair_guidance():
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "10:00-11:30",
                    "activity": "亲子小小科学实验室",
                    "poi_id": "act_parent_science_lab",
                },
                {
                    "time": "18:30-20:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                },
            ]
        },
        "tool_results": {
            "order_activity_ticket_1": {
                "success": False,
                "name": "亲子小小科学实验室",
                "failure_reason": "merchant_closed",
            }
        },
        "action_sequence": [
            {
                "action_type": "order_activity_ticket",
                "poi_id": "act_parent_science_lab",
                "time": "10:00",
            }
        ],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "b_repair_plan": {
            "repair_strategy": "retry_same_poi_new_slot",
            "user_message": "10:00时段商家关闭，可以切换到14:30可用时段。",
        },
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "亲子小小科学实验室" in message
    assert "调整建议" in message
    assert "14:30可用时段" in message


def test_partial_restaurant_failure_preserves_low_wait_rain_constraints():
    state = {
        "scene_type": "friends",
        "constraints": {
            "avoid": ["排队久", "人多拥挤"],
            "raw_text": "可能下雨，不想排队，也不想跑来跑去。",
        },
        "scenario_facets": {
            "risk_posture": ["排队久"],
        },
        "selected_plan": {
            "timeline": [
                {
                    "time": "14:00-16:00",
                    "activity": "上海市历史博物馆",
                    "poi_id": "gaode_act_B0FFI2885X",
                },
                {
                    "time": "17:30-18:50",
                    "activity": "鮨士道寿司轻食(日月光中心店)",
                    "poi_id": "gaode_res_B0J1GD89GK",
                },
            ]
        },
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "name": "上海市历史博物馆",
                "data": {"action": "order_activity_ticket"},
            },
            "reserve_restaurant_2": {
                "success": False,
                "name": "鮨士道寿司轻食(日月光中心店)",
                "data": {
                    "action": "reserve_restaurant",
                    "failure_reason": "slot_full",
                },
            },
        },
        "execution_status": "partial",
        "payment_status": "not_required",
        "execution_log": [],
        "b_repair_plan": {
            "repair_strategy": "replace_failed_node",
            "user_message": "建议换一家可预约且等待更短的不辣餐厅。",
        },
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "到了现场再看" not in message
    assert "现场等位" in message
    assert "雨天和排队风险" in message
    assert "调整建议" in message
    assert "等待更短" in message
