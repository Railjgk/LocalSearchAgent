from src.nodes.tool_router import tool_router_node


def _partial_state(missing_roles: list[str]) -> dict:
    return {
        "scene_type": "solo",
        "constraints": {"people_count": 2},
        "selected_plan": {
            "plan_id": "partial_guidance_plan",
            "execution_ready": False,
            "execution_scope": "partial",
            "partial_missing_roles": missing_roles,
            "timeline": [
                {
                    "poi_id": "res_001",
                    "activity": "Test Restaurant",
                    "type": "restaurant",
                    "time": "18:30-19:30",
                }
            ],
            "action_hints": [
                {
                    "action_type": "reserve_restaurant",
                    "poi_id": "res_001",
                    "time": "18:30",
                    "name": "Test Restaurant",
                }
            ],
        },
        "execution_log": [],
    }


def test_tool_router_executes_supported_actions_when_missing_roles_are_guidance_only():
    result = tool_router_node(_partial_state(["convenience_store", "souvenir_shopping"]))

    assert result["action_sequence"]
    assert result["action_sequence"][0]["action_type"] == "reserve_restaurant"
    assert any("guidance-only" in item for item in result["execution_log"])


def test_tool_router_blocks_partial_plan_when_lodging_is_missing():
    result = tool_router_node(_partial_state(["lodging"]))

    assert result["action_sequence"] == []
    assert any("blocking missing itinerary roles" in item for item in result["execution_log"])
