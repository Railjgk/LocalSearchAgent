from __future__ import annotations

import json

from src.nodes import b_repair_planner


def _clear_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "WF_B_AI_REPAIR_PLANNER_ENABLED",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _failed_state() -> dict:
    return {
        "user_input": "今天下午带老婆孩子出去玩，别太折腾。",
        "scene_type": "family",
        "constraints": {"people_count": 3, "child_age": 5, "max_distance_km": 8},
        "user_profile": {"avoid": ["long_queue"]},
        "scenario_activities": ["亲子", "轻食"],
        "selected_plan": {
            "plan_id": "plan_main",
            "title": "亲子下午计划",
            "scene_type": "family",
            "total_price": 360,
            "total_distance_km": 4.8,
            "timeline": [
                {"type": "play", "activity": "树屋亲子探索馆", "poi_id": "act_tree", "time": "15:00-16:30"},
                {"type": "restaurant", "activity": "禾间轻食餐厅", "poi_id": "res_light", "time": "17:30-18:40"},
            ],
            "action_hints": [
                {"step": 1, "action_type": "order_activity_ticket", "poi_id": "act_tree", "time": "15:00"},
                {"step": 2, "action_type": "reserve_restaurant", "poi_id": "res_light", "time": "17:30"},
            ],
        },
        "alternative_plans": [
            {"plan_id": "plan_alt_route", "title": "路线更短", "dominant_dimension": "route", "tradeoff": "更近但氛围弱"},
            {"plan_id": "plan_alt_budget", "title": "预算更低", "dominant_dimension": "budget", "tradeoff": "更便宜但距离略远"},
        ],
        "action_sequence": [
            {"step": 1, "action_type": "order_activity_ticket", "poi_id": "act_tree", "time": "15:00"},
            {"step": 2, "action_type": "reserve_restaurant", "poi_id": "res_light", "time": "17:30"},
        ],
        "execution_commit_result": {
            "success": False,
            "overall_status": "failed",
            "failed_step": "reserve_restaurant_2",
            "failure_reason": "slot_full",
            "retry_history": [],
            "steps": [
                {
                    "step_id": "order_activity_ticket_1",
                    "action_type": "order_activity_ticket",
                    "success": True,
                    "status": "ordered",
                    "poi_id": "act_tree",
                    "time": "15:00",
                },
                {
                    "step_id": "reserve_restaurant_2",
                    "action_type": "reserve_restaurant",
                    "success": False,
                    "status": "failed",
                    "failure_reason": "slot_full",
                    "poi_id": "res_light",
                    "time": "17:30",
                    "alternatives": [{"time": "18:30", "remaining": 6}],
                },
            ],
        },
        "execution_status": "failed",
        "execution_log": [],
    }


def test_repair_planner_is_default_off(monkeypatch):
    _clear_env(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("repair planner should be default off")

    monkeypatch.setattr(b_repair_planner, "chat_completion", fail_if_called)

    assert b_repair_planner.repair_planner_node(_failed_state()) == {}


def test_repair_planner_skips_success_even_when_enabled(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("repair planner should not run after success")

    monkeypatch.setattr(b_repair_planner, "chat_completion", fail_if_called)
    state = _failed_state()
    state["execution_status"] = "success"
    state["execution_commit_result"] = {"success": True, "overall_status": "completed", "steps": []}

    assert b_repair_planner.repair_planner_node(state) == {}


def test_repair_planner_uses_c_alternatives_and_filters_inventions(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    captured = {}

    def fake_chat_completion(messages, *, config):
        captured["payload"] = json.loads(messages[1]["content"])
        return {
            "content": json.dumps(
                {
                    "repair_strategy": "retry_same_poi_new_slot",
                    "preserve_poi_ids": ["act_tree", "invented_poi"],
                    "replace_failed_node": {
                        "failed_poi_id": "res_light",
                        "preferred_category": "轻食餐厅",
                        "constraints": ["儿童友好", "低卡"],
                        "reason": "保留亲子活动，只调整餐厅履约方式",
                    },
                    "time_adjustments": [
                        {"action_step": 2, "from_time": "17:30", "to_time": "18:30", "reason": "C 返回该时段可用"},
                        {"action_step": 2, "from_time": "17:30", "to_time": "22:00", "reason": "invented"},
                    ],
                    "candidate_plan_ids": ["plan_alt_route", "invented_plan"],
                    "user_message": "17:30 已满，可以保留活动，把餐厅改到 18:30。",
                    "confidence": 0.82,
                    "evidence": ["C 返回 18:30 可用"],
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 77},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_repair_planner, "chat_completion", fake_chat_completion)
    result = b_repair_planner.repair_planner_node(_failed_state())

    payload = captured["payload"]
    assert payload["failure_context"]["commit"]["failure_reason"] == "slot_full"
    assert "18:30" in payload["guardrails"]["allowed_alternative_times"]
    assert result["b_repair_plan"]["repair_strategy"] == "retry_same_poi_new_slot"
    assert result["b_repair_plan"]["preserve_poi_ids"] == ["act_tree"]
    assert result["b_repair_plan"]["time_adjustments"] == [
        {"action_step": 2, "from_time": "17:30", "to_time": "18:30", "reason": "C 返回该时段可用"}
    ]
    assert result["b_repair_plan"]["candidate_plan_ids"] == ["plan_alt_route"]
    assert result["b_ai_repair_plan"]["success"] is True
    assert "test-key" not in str(result)


def test_repair_planner_sanitizes_false_per_person_budget_claim(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "repair_strategy": "switch_to_alternative_plan",
                    "candidate_plan_ids": ["plan_alt_budget"],
                    "user_message": (
                        "当前选定的方案总费用约330元，超出人均200元预算。"
                        "建议切换到低预算备选方案。"
                    ),
                    "confidence": 0.8,
                    "evidence": ["总费用330元，人均82.5元，未超预算"],
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 88},
            "finish_reason": "stop",
        }

    state = _failed_state()
    state["constraints"].update(
        {
            "budget": 200,
            "budget_type": "per_person",
            "people_count": 4,
            "raw_text": "我们四个朋友预算人均200左右。",
        }
    )
    state["selected_plan"]["total_price"] = 330
    monkeypatch.setattr(b_repair_planner, "chat_completion", fake_chat_completion)

    result = b_repair_planner.repair_planner_node(state)

    message = result["b_repair_plan"]["user_message"]
    assert "未超出人均200元预算" in message
    assert "总费用约330元，超出" not in message
    assert "82.5元/人" in message


def test_repair_planner_fallback_redacts_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        raise RuntimeError("bad key test-key")

    monkeypatch.setattr(b_repair_planner, "chat_completion", fake_chat_completion)
    result = b_repair_planner.repair_planner_node(_failed_state())

    assert "b_repair_plan" not in result
    assert result["b_ai_repair_plan"]["success"] is False
    assert result["b_ai_repair_plan"]["fallback"] is True
    assert "test-key" not in str(result)
