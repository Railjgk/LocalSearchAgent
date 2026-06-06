from src.nodes.share_generator import share_generator_node


def test_failed_share_reports_non_executable_plan_without_system_busy():
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:30",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                }
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "系统繁忙" not in message
    assert "没有形成可执行的预订动作" in message
    assert "禾间轻食日料" in message


def test_failed_non_executable_share_includes_repair_guidance():
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:15",
                    "activity": "云栖温泉轻餐茶室",
                    "poi_id": "res_spa_light_tea",
                }
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
        "b_repair_plan": {
            "repair_strategy": "ask_user_confirm",
            "user_message": "当前方案仅匹配到一个午餐节点，缺少上午和下午的活动节点，请确认是否重新规划完整行程。",
        },
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "没有形成可执行的预订动作" in message
    assert "调整建议" in message
    assert "缺少上午和下午的活动节点" in message


def test_failed_skeleton_share_names_non_executable_roles():
    state = {
        "scene_type": "couple",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:10",
                    "activity": "餐饮",
                    "poi_id": None,
                    "role": "restaurant_lunch",
                    "type": "planning_intent",
                },
                {
                    "time": "15:00-次日10:00",
                    "activity": "住宿",
                    "poi_id": None,
                    "role": "lodging",
                    "type": "planning_intent",
                },
                {
                    "time": "15:45-16:00",
                    "activity": "停车",
                    "poi_id": None,
                    "role": "parking",
                    "type": "planning_intent",
                },
            ],
        },
        "b_itinerary_blueprint": {
            "unsupported_roles": ["lodging", "parking"],
            "node_intents": [
                {"role": "lodging", "label": "住宿"},
                {"role": "parking", "label": "停车"},
            ],
        },
        "b_rag_candidate_coverage": {
            "unsupported_missing_roles": ["lodging", "parking"],
            "missing_node_ids": ["intent_02", "intent_03"],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "没有形成可执行的预订动作" in message
    assert "还只是行程意图" in message
    assert "住宿" in message
    assert "停车" in message
    assert "不能说已经订好" in message


def test_success_share_still_reports_unsupported_dropped_role():
    state = {
        "scene_type": "solo",
        "constraints": {
            "people_count": 1,
            "raw_text": "想先找个20点以后还能处理的口腔/牙科看看，附近再吃点不辣、软一点的晚饭。",
        },
        "selected_plan": {
            "timeline": [
                {
                    "time": "18:30-20:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                    "role": "restaurant_dinner",
                }
            ],
        },
        "b_itinerary_blueprint": {
            "unsupported_roles": ["dental_clinic"],
            "node_intents": [
                {"role": "dental_clinic", "label": "牙科/口腔诊所"},
                {"role": "restaurant_dinner", "label": "晚餐"},
            ],
        },
        "b_rag_candidate_coverage": {
            "unsupported_missing_roles": ["dental_clinic"],
            "missing_node_ids": ["intent_01"],
        },
        "tool_results": {
            "reserve_restaurant_1": {
                "success": True,
                "name": "禾间轻食日料",
                "data": {"action": "reserve_restaurant"},
            }
        },
        "execution_status": "success",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "可执行的部分已经订好了" in message
    assert "禾间轻食日料" in message
    assert "牙科/口腔诊所" in message
    assert "不能说已经订好" in message
