from src.nodes.share_generator import share_generator_node


def test_share_generator_uses_execution_retry_time_when_timeline_stays_ordered() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "activity": "上海市历史博物馆",
                    "poi_id": "gaode_act_B0FFI2885X",
                    "time": "14:00-16:00",
                },
                {
                    "activity": "附近休息与转场",
                    "poi_id": None,
                    "time": "16:00-16:30",
                },
                {
                    "activity": "熹誉海鲜火锅(大上海时代广场店)",
                    "poi_id": "gaode_res_B0H17H7L5G",
                    "time": "17:30-19:30",
                },
            ]
        },
        "tool_results": {
            "reserve_restaurant_2": {
                "success": True,
                "name": "熹誉海鲜火锅(大上海时代广场店)",
                "data": {
                    "time": "18:00",
                    "raw_step": {
                        "poi_id": "gaode_res_B0H17H7L5G",
                        "time": "18:00",
                    },
                },
            }
        },
        "execution_status": "success",
        "scene_type": "friends",
        "execution_log": [],
    }

    result = share_generator_node(state)

    message = result["final_share_message"]
    assert "18:00-20:00 熹誉海鲜火锅(大上海时代广场店)" in message
    assert "17:30-19:30 熹誉海鲜火锅(大上海时代广场店)" not in message


def test_share_generator_keeps_plan_time_when_execution_time_would_overlap() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "activity": "上海市历史博物馆",
                    "poi_id": "gaode_act_B0FFI2885X",
                    "time": "17:00-19:00",
                },
                {
                    "activity": "附近休息与转场",
                    "poi_id": None,
                    "time": "19:00-19:30",
                },
                {
                    "activity": "熹誉海鲜火锅(大上海时代广场店)",
                    "poi_id": "gaode_res_B0H17H7L5G",
                    "time": "19:30-21:30",
                },
            ]
        },
        "tool_results": {
            "reserve_restaurant_2": {
                "success": True,
                "name": "熹誉海鲜火锅(大上海时代广场店)",
                "data": {
                    "time": "17:30",
                    "raw_step": {
                        "poi_id": "gaode_res_B0H17H7L5G",
                        "time": "17:30",
                    },
                },
            }
        },
        "execution_status": "success",
        "scene_type": "friends",
        "execution_log": [],
    }

    result = share_generator_node(state)

    message = result["final_share_message"]
    assert "19:00-19:30 附近休息与转场" in message
    assert "19:30-21:30 熹誉海鲜火锅(大上海时代广场店)" in message
    assert "17:30-19:30 熹誉海鲜火锅(大上海时代广场店)" not in message
