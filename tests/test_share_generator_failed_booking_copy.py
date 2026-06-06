from src.nodes.share_generator import share_generator_node


def test_failed_booking_share_lists_timeline_and_all_failed_items() -> None:
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
            ],
        },
        "tool_results": {
            "order_activity_ticket_1": {
                "success": False,
                "name": "树屋亲子陶艺体验馆",
                "failure_reason": "inventory_empty",
            },
            "reserve_restaurant_2": {
                "success": False,
                "name": "禾间轻食日料",
                "failure_reason": "inventory_empty",
            },
        },
        "action_sequence": [
            {
                "action_type": "order_activity_ticket",
                "poi_id": "act_family_ceramic",
                "time": "15:30",
            },
            {
                "action_type": "reserve_restaurant",
                "poi_id": "res_light_japanese",
                "time": "17:30",
            },
        ],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "15:30-17:00 树屋亲子陶艺体验馆" in message
    assert "17:30-19:00 禾间轻食日料" in message
    assert "未订上：树屋亲子陶艺体验馆、禾间轻食日料" in message


def test_failed_booking_repair_guidance_does_not_claim_unexecuted_replacement() -> None:
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "18:00-19:30",
                    "activity": "月光小酒馆轻餐",
                    "poi_id": "res_romantic_bistro",
                },
                {
                    "time": "20:00-22:00",
                    "activity": "上海自然博物馆",
                    "poi_id": "gaode_act_B00156NVZG",
                },
            ],
        },
        "tool_results": {
            "reserve_restaurant_1": {
                "success": False,
                "name": "月光小酒馆轻餐",
                "failure_reason": "inventory_empty",
            },
            "order_activity_ticket_2": {
                "success": False,
                "name": "上海自然博物馆",
                "failure_reason": "inventory_empty",
            },
        },
        "action_sequence": [
            {
                "action_type": "reserve_restaurant",
                "poi_id": "res_romantic_bistro",
                "time": "18:00",
            },
            {
                "action_type": "order_activity_ticket",
                "poi_id": "gaode_act_B00156NVZG",
                "time": "20:00",
            },
        ],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "b_repair_plan": {
            "repair_strategy": "replace_failed_node",
            "user_message": "周六晚餐时段原餐厅已满座，已为您替换为同类型可预约的轻食/健康餐厅，并调整到18:30开始。博物馆/展览节点也已同步替换为同类型可预约场馆，请稍候。",
        },
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "调整建议" in message
    assert "已为您替换" not in message
    assert "已同步替换" not in message
    assert "请稍候" not in message
    assert "建议替换为同类型可预约的轻食/健康餐厅" in message
    assert "建议替换为同类型可预约场馆" in message


def test_family_partial_share_does_not_overclaim_minority_success() -> None:
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "14:30-16:00",
                    "activity": "亲子小小科学实验室",
                    "poi_id": "act_parent_science_lab",
                },
                {
                    "time": "17:00-18:30",
                    "activity": "树屋亲子陶艺体验馆",
                    "poi_id": "act_family_ceramic",
                },
                {
                    "time": "19:30-21:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                },
                {
                    "time": "21:30-22:30",
                    "activity": "青禾沙拉碗",
                    "poi_id": "res_salad_bowl",
                },
            ],
        },
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "name": "亲子小小科学实验室",
                "data": {"action": "order_activity_ticket"},
            },
            "order_activity_ticket_2": {
                "success": False,
                "name": "树屋亲子陶艺体验馆",
                "failure_reason": "inventory_empty",
            },
            "reserve_restaurant_3": {
                "success": False,
                "name": "禾间轻食日料",
                "failure_reason": "inventory_empty",
            },
            "reserve_restaurant_4": {
                "success": False,
                "name": "青禾沙拉碗",
                "failure_reason": "inventory_empty",
            },
        },
        "execution_status": "partial",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "大部分安排好了" not in message
    assert "目前只确认了亲子小小科学实验室" in message
    assert "树屋亲子陶艺体验馆、禾间轻食日料、青禾沙拉碗还没订上" in message
