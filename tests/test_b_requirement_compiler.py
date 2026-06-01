from __future__ import annotations

import json

from src.nodes import b_requirement_compiler
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node


def _clear_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "WF_B_AI_REQUIREMENT_COMPILER_ENABLED",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _plan(plan_id: str, activity: dict, restaurant: dict) -> dict:
    return {
        "plan_id": plan_id,
        "nodes": [
            {
                "poi_id": f"act_{plan_id}",
                "type": "activity",
                "name": "活动",
                "tags": [],
                "available": True,
                **activity,
            },
            {
                "poi_id": f"res_{plan_id}",
                "type": "restaurant",
                "name": "餐厅",
                "tags": [],
                "available": True,
                "dine_in_available": True,
                **restaurant,
            },
        ],
        "route": {"total_distance_km": 2.0, "total_travel_time_min": 20},
        "budget": {"total_price": 300},
        "availability": {"all_available": True, "max_queue_time_min": 10},
        "estimated_duration_min": 240,
        "tags": [],
    }


def test_deterministic_requirement_contract_detects_product_guardrails(monkeypatch):
    _clear_env(monkeypatch)

    constraints = {
        "child_age": 5,
        "people_count": 3,
        "planning_preferences": {
            "food_type": ["咖啡"],
            "activity_type": ["看展"],
        },
    }
    state = {
        "user_input": "今天下午和孩子看展，再找个咖啡店坐一会儿，不想吃正餐。",
        "scene_type": "family",
        "constraints": constraints,
    }

    enhanced, contract, metadata = b_requirement_compiler.apply_b_requirement_contract(
        state,
        constraints=constraints,
    )

    assert metadata is None
    assert "child_friendly_activity" in contract["hard_requirements"]
    assert "cafe_non_full_meal" in contract["hard_requirements"]
    assert "正餐" in contract["forbidden_restaurant_groups"]
    assert "亲子" in enhanced["planning_preferences"]["activity_type"]
    assert "咖啡" in enhanced["planning_preferences"]["food_type"]


def test_requirement_compiler_uses_longcat_when_enabled(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        payload = json.loads(messages[1]["content"])
        assert "deterministic_contract" in payload
        return {
            "content": json.dumps(
                {
                    "hard_requirements": ["pet_friendly", "invented"],
                    "needs_confirmation": ["宠物政策需要确认"],
                    "evidence": ["用户说带狗"],
                },
                ensure_ascii=False,
            ),
            "model": config.model,
            "usage": {"total_tokens": 31},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_requirement_compiler, "chat_completion", fake_chat_completion)
    _, contract, metadata = b_requirement_compiler.apply_b_requirement_contract(
        {"user_input": "带狗找个地方玩", "scene_type": "friends", "constraints": {}},
        constraints={},
    )

    assert "pet_friendly" in contract["hard_requirements"]
    assert metadata["success"] is True
    assert metadata["usage"] == {"total_tokens": 31}
    assert "test-key" not in str(metadata)


def test_constraint_filter_rejects_child_plan_without_child_evidence():
    state = {
        "constraints": {
            "child_age": 5,
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": ["child_friendly_activity"],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            _plan(
                "spa_plan",
                {"name": "波比SPA", "tags": ["低强度", "放松"], "category": "近场放松"},
                {"name": "米禾良日料", "tags": ["轻食"], "restaurant_category": "日料轻食"},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["spa_plan"] == "缺少明确儿童友好/亲子活动证据"


def test_constraint_filter_rejects_full_meal_for_cafe_nonmeal_intent():
    contract = {
        "hard_requirements": ["cafe_non_full_meal"],
        "forbidden_restaurant_groups": ["正餐"],
    }
    common = {
        "budget": 600,
        "max_distance_km": 8,
        "max_queue_time_min": 30,
        "duration_range": [3, 5],
        "b_requirement_contract": contract,
    }
    state = {
        "constraints": common,
        "candidates": [
            _plan(
                "hotpot_plan",
                {"name": "上海城市规划展示馆", "tags": ["博物馆展览"], "category": "博物馆展览"},
                {"name": "巴奴毛肚火锅", "tags": ["火锅"], "restaurant_category": "火锅"},
            ),
            _plan(
                "coffee_plan",
                {"name": "上海城市规划展示馆", "tags": ["博物馆展览"], "category": "博物馆展览"},
                {"name": "石藤咖啡", "tags": ["咖啡"], "restaurant_category": "咖啡甜品"},
            ),
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["coffee_plan"]
    assert result["filter_reasons"]["hotpot_plan"] == "餐厅命中用户明确规避的正餐需求"


def test_candidate_generator_attaches_requirement_contract(monkeypatch):
    _clear_env(monkeypatch)
    result = candidate_generator_node(
        {
            "user_input": "今天下午想和老婆孩子出去玩，孩子5岁。",
            "scene_type": "family",
            "constraints": {"child_age": 5, "people_count": 3},
            "user_profile": {},
            "scenario_activities": [],
            "execution_log": [],
        }
    )

    assert result["b_requirement_contract"]["hard_requirements"] == ["child_friendly_activity"]
    assert "b_requirement_contract" in result["constraints"]
