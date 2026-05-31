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
                "name": "\u6d3b\u52a8",
                "tags": [],
                "available": True,
                **activity,
            },
            {
                "poi_id": f"res_{plan_id}",
                "type": "restaurant",
                "name": "\u9910\u5385",
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
            "food_type": ["\u5496\u5561"],
            "activity_type": ["\u770b\u5c55"],
        },
    }
    state = {
        "user_input": (
            "\u4eca\u5929\u4e0b\u5348\u548c\u5b69\u5b50\u770b\u5c55\uff0c"
            "\u518d\u627e\u4e2a\u5496\u5561\u5e97\u5750\u4e00\u4f1a\u513f\uff0c"
            "\u4e0d\u60f3\u5403\u6b63\u9910\u3002"
        ),
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
    assert "\u6b63\u9910" in contract["forbidden_restaurant_groups"]
    assert "\u4eb2\u5b50" in enhanced["planning_preferences"]["activity_type"]
    assert "\u5496\u5561" in enhanced["planning_preferences"]["food_type"]


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
                    "needs_confirmation": ["\u5ba0\u7269\u653f\u7b56\u9700\u8981\u786e\u8ba4"],
                    "evidence": ["\u7528\u6237\u8bf4\u5e26\u72d7"],
                },
                ensure_ascii=False,
            ),
            "model": config.model,
            "usage": {"total_tokens": 31},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_requirement_compiler, "chat_completion", fake_chat_completion)
    _, contract, metadata = b_requirement_compiler.apply_b_requirement_contract(
        {"user_input": "\u5e26\u72d7\u627e\u4e2a\u5730\u65b9\u73a9", "scene_type": "friends", "constraints": {}},
        constraints={},
    )

    assert "pet_friendly" in contract["hard_requirements"]
    assert metadata["success"] is True
    assert metadata["usage"] == {"total_tokens": 31}
    assert "test-key" not in str(metadata)


def test_requirement_compiler_can_skip_longcat_for_fast_rag_retrieval(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fail_if_called(messages, *, config):
        raise AssertionError("LongCat should not be called when allow_llm=False")

    monkeypatch.setattr(b_requirement_compiler, "chat_completion", fail_if_called)
    _, contract, metadata = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": "\u5e26\u5b69\u5b50\u627e\u5ba4\u5185\u4eb2\u5b50\u6d3b\u52a8",
            "scene_type": "family",
            "constraints": {"child_age": 5, "people_count": 3},
        },
        constraints={"child_age": 5, "people_count": 3},
        allow_llm=False,
    )

    assert "child_friendly_activity" in contract["hard_requirements"]
    assert metadata["skipped"] is True
    assert metadata["reason"] == "deterministic_only"


def test_requirement_compiler_does_not_treat_adult_party_count_as_child_age(monkeypatch):
    _clear_env(monkeypatch)
    _, contract, _ = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": "\u6211\u4eec4\u4e2a\u4eba\u60f3\u627e\u4e2a\u9ad8\u6863\u70b9\u7684\u6252\u623f\u5e86\u795d",
            "scene_type": "friends",
            "constraints": {"child_age": 4, "people_count": 4},
        },
        constraints={"child_age": 4, "people_count": 4},
        allow_llm=False,
    )

    assert "child_friendly_activity" not in contract["hard_requirements"]
    assert "restaurant_reservation" in contract["hard_requirements"]


def test_requirement_compiler_rejects_unsupported_longcat_hard_guards(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "hard_requirements": ["cafe_non_full_meal", "parking_needed"],
                    "forbidden_restaurant_groups": ["\u6b63\u9910"],
                    "evidence": ["model overreach"],
                },
                ensure_ascii=False,
            ),
            "model": config.model,
            "usage": {"total_tokens": 29},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_requirement_compiler, "chat_completion", fake_chat_completion)
    enhanced, contract, metadata = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": "friends want lamb skewers and a relaxed activity",
            "scene_type": "friends",
            "constraints": {"planning_preferences": {"food_type": ["\u7f8a\u8089\u4e32"]}},
        },
        constraints={"planning_preferences": {"food_type": ["\u7f8a\u8089\u4e32"]}},
    )

    assert metadata["success"] is True
    assert "cafe_non_full_meal" not in contract["hard_requirements"]
    assert "parking_needed" not in contract["hard_requirements"]
    assert "\u6b63\u9910" not in contract["forbidden_restaurant_groups"]
    assert enhanced["planning_preferences"]["food_type"] == ["\u7f8a\u8089\u4e32"]


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
                {"name": "Bobby SPA", "tags": ["\u4f4e\u5f3a\u5ea6", "\u653e\u677e"], "category": "\u8fd1\u573a\u653e\u677e"},
                {"name": "Mitori", "tags": ["\u8f7b\u98df"], "restaurant_category": "\u65e5\u6599\u8f7b\u98df"},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert "spa_plan" in result["filter_reasons"]


def test_constraint_filter_does_not_apply_child_age_without_child_context():
    state = {
        "constraints": {
            "raw_text": "\u6211\u4eec4\u4e2a\u4eba\u60f3\u627e\u6252\u623f\u5e86\u795d",
            "scene": "friends",
            "child_age": 4,
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            _plan(
                "adult_steak_plan",
                {"name": "\u5916\u6ee9\u6563\u6b65", "tags": ["\u57ce\u5e02\u6f2b\u6b65"]},
                {"name": "\u7ea2\u5c4b\u725b\u6392\u9601", "tags": ["\u725b\u6392", "\u897f\u9910"]},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["adult_steak_plan"]


def test_constraint_filter_does_not_apply_spouse_diet_memory_without_spouse_context():
    state = {
        "constraints": {
            "raw_text": "\u6211\u4eec4\u4e2a\u4eba\u60f3\u627e\u6252\u623f\u5e86\u795d",
            "scene": "friends",
            "mom_diet": None,
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "user_profile": {
            "companion_profile": {
                "wife": {"state": "dieting", "needs": ["low_calorie", "light_food"]}
            }
        },
        "candidates": [
            _plan(
                "adult_steak_plan",
                {"name": "\u5916\u6ee9\u6563\u6b65", "tags": ["\u57ce\u5e02\u6f2b\u6b65"]},
                {"name": "\u7ea2\u5c4b\u725b\u6392\u9601", "tags": ["\u725b\u6392", "\u897f\u9910"]},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["adult_steak_plan"]


def test_constraint_filter_keeps_romantic_dinner_when_low_calorie_is_only_soft_memory():
    state = {
        "constraints": {
            "raw_text": "\u60c5\u4eba\u8282\u5e26\u5973\u670b\u53cb\u5403\u4e2a\u6d6a\u6f2b\u7684\u665a\u9910",
            "scene": "friends",
            "mom_diet": "low_calorie",
            "soft_tags": ["\u4f4e\u5361", "\u8f7b\u98df"],
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "user_profile": {
            "companion_profile": {
                "wife": {"state": "dieting", "needs": ["low_calorie", "light_food"]}
            }
        },
        "candidates": [
            _plan(
                "romantic_steak_plan",
                {"name": "\u6c5f\u666f\u6563\u6b65", "tags": ["\u6c5f\u666f"]},
                {"name": "\u6d66\u6c5f\u897f\u9910\u5385", "tags": ["\u897f\u9910", "\u6c5f\u666f"]},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["romantic_steak_plan"]


def test_constraint_filter_applies_wife_diet_memory_when_wife_is_current_companion():
    state = {
        "constraints": {
            "raw_text": "\u4eca\u5929\u548c\u8001\u5a46\u5403\u665a\u9910",
            "scene": "couple",
            "mom_diet": "low_calorie",
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "user_profile": {
            "companion_profile": {
                "wife": {"state": "dieting", "needs": ["low_calorie", "light_food"]}
            }
        },
        "candidates": [
            _plan(
                "heavy_steak_plan",
                {"name": "\u6c5f\u666f\u6563\u6b65", "tags": ["\u6c5f\u666f"]},
                {"name": "\u6d66\u6c5f\u897f\u9910\u5385", "tags": ["\u897f\u9910", "\u725b\u6392"]},
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert "heavy_steak_plan" in result["filter_reasons"]


def test_constraint_filter_rejects_full_meal_for_cafe_nonmeal_intent():
    contract = {
        "hard_requirements": ["cafe_non_full_meal"],
        "forbidden_restaurant_groups": ["\u6b63\u9910"],
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
                {"name": "\u57ce\u5e02\u89c4\u5212\u5c55\u793a\u9986", "tags": ["\u535a\u7269\u9986\u5c55\u89c8"], "category": "\u535a\u7269\u9986\u5c55\u89c8"},
                {"name": "\u5df4\u5974\u6bdb\u809a\u706b\u9505", "tags": ["\u706b\u9505"], "restaurant_category": "\u706b\u9505"},
            ),
            _plan(
                "coffee_plan",
                {"name": "\u57ce\u5e02\u89c4\u5212\u5c55\u793a\u9986", "tags": ["\u535a\u7269\u9986\u5c55\u89c8"], "category": "\u535a\u7269\u9986\u5c55\u89c8"},
                {"name": "\u77f3\u85e4\u5496\u5561", "tags": ["\u5496\u5561"], "restaurant_category": "\u5496\u5561\u751c\u54c1"},
            ),
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["coffee_plan"]
    assert "hotpot_plan" in result["filter_reasons"]


def test_forbidden_barbecue_uses_primary_evidence_not_generic_meat_tag():
    common = {
        "budget": 600,
        "max_distance_km": 8,
        "max_queue_time_min": 30,
        "duration_range": [3, 5],
        "b_requirement_contract": {
            "hard_requirements": [],
            "forbidden_restaurant_groups": ["\u70e4\u8089"],
        },
    }
    state = {
        "constraints": common,
        "candidates": [
            _plan(
                "hotpot_with_meat_tag",
                {"name": "\u6d77\u6d3e\u624b\u4f5c\u9986", "tags": ["\u624b\u4f5c"]},
                {
                    "name": "\u5df4\u5974\u6bdb\u809a\u706b\u9505",
                    "tags": ["\u706b\u9505", "meat"],
                    "restaurant_category": "\u706b\u9505",
                },
            ),
            _plan(
                "bbq_plan",
                {"name": "\u6d77\u6d3e\u624b\u4f5c\u9986", "tags": ["\u624b\u4f5c"]},
                {
                    "name": "\u70ad\u706b\u70e4\u8089",
                    "tags": ["meat"],
                    "restaurant_category": "\u70e4\u8089",
                },
            ),
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["hotpot_with_meat_tag"]
    assert "bbq_plan" in result["filter_reasons"]


def test_constraint_filter_allows_multiday_partial_plan_with_parking_node():
    state = {
        "constraints": {
            "budget": 2000,
            "max_distance_km": 30,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": ["parking_needed"],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "overnight_with_parking",
                "planner_mode": "multi_node_itinerary",
                "execution_scope": "partial",
                "planning_days": 2,
                "planning_horizon": "overnight",
                "nodes": [
                    {
                        "poi_id": "hotel_1",
                        "type": "hotel",
                        "itinerary_role": "lodging",
                        "name": "\u5916\u6ee9\u5bb6\u5ead\u516c\u5bd3",
                        "available": True,
                    },
                    {
                        "poi_id": "res_1",
                        "type": "restaurant",
                        "name": "\u87f9\u9ec4\u9762",
                        "available": True,
                        "dine_in_available": True,
                    },
                    {
                        "poi_id": "parking_1",
                        "type": "transport_service",
                        "itinerary_role": "parking",
                        "name": "\u5916\u6ee9\u505c\u8f66\u573a",
                        "available": True,
                    },
                ],
                "route": {"total_distance_km": 6, "total_travel_time_min": 35},
                "budget": {"total_price": 760},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["overnight_with_parking"]


def test_constraint_filter_uses_leg_distance_for_multi_node_itinerary():
    state = {
        "constraints": {
            "budget": 2000,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "multi_node_nearby_legs",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "full_day",
                "nodes": [
                    {"poi_id": "hotel_1", "type": "hotel", "itinerary_role": "lodging", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                    {"poi_id": "shop_1", "type": "shopping", "itinerary_role": "convenience_store", "available": True},
                    {"poi_id": "park_1", "type": "transport_service", "itinerary_role": "parking", "available": True},
                ],
                "route": {
                    "total_distance_km": 18,
                    "total_travel_time_min": 80,
                    "legs": [
                        {"distance_km": 6},
                        {"distance_km": 6},
                        {"distance_km": 6},
                    ],
                },
                "budget": {"total_price": 760},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["multi_node_nearby_legs"]


def test_constraint_filter_does_not_apply_pair_lower_duration_to_multi_node():
    state = {
        "constraints": {
            "budget": 1200,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 360],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "short_multi_node",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                    {"poi_id": "shop_1", "type": "shopping", "itinerary_role": "convenience_store", "available": True},
                    {"poi_id": "ktv_1", "type": "activity", "itinerary_role": "karaoke", "available": True},
                ],
                "route": {
                    "total_distance_km": 5,
                    "total_travel_time_min": 35,
                    "legs": [{"distance_km": 2}, {"distance_km": 3}],
                },
                "budget": {"total_price": 500},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 150,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["short_multi_node"]


def test_constraint_filter_does_not_apply_default_upper_duration_to_multi_node():
    state = {
        "constraints": {
            "raw_text": "\u805a\u9910\u540e\u8fd8\u8981\u53bbKTV\u5531\u6b4c",
            "budget": 1200,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 360],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "long_multi_node_without_explicit_duration",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                    {"poi_id": "ktv_1", "type": "activity", "itinerary_role": "karaoke", "available": True},
                ],
                "route": {
                    "total_distance_km": 4,
                    "total_travel_time_min": 30,
                    "legs": [{"distance_km": 4}],
                },
                "budget": {"total_price": 500},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 480,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["long_multi_node_without_explicit_duration"]


def test_constraint_filter_does_not_hard_reject_default_budget_for_lodging():
    state = {
        "constraints": {
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "overnight_without_explicit_budget",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 2,
                "planning_horizon": "overnight",
                "nodes": [
                    {"poi_id": "hotel_1", "type": "hotel", "itinerary_role": "lodging", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                ],
                "route": {
                    "total_distance_km": 6,
                    "total_travel_time_min": 35,
                    "legs": [{"distance_km": 6}],
                },
                "budget": {"total_price": 900},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["overnight_without_explicit_budget"]


def test_constraint_filter_treats_vague_cheap_lodging_budget_as_soft():
    state = {
        "constraints": {
            "raw_text": "\u60f3\u627e\u4e2a\u4fbf\u5b9c\u70b9\u7684\u5730\u65b9\u4f4f\u4e00\u665a",
            "budget": 300,
            "budget_type": "total",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "cheap_vague_lodging",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 2,
                "planning_horizon": "overnight",
                "nodes": [
                    {"poi_id": "hotel_1", "type": "hotel", "itinerary_role": "lodging", "available": True},
                    {"poi_id": "cafe_1", "type": "restaurant", "itinerary_role": "cafe", "available": True},
                ],
                "route": {
                    "total_distance_km": 3,
                    "total_travel_time_min": 25,
                    "legs": [{"distance_km": 3}],
                },
                "budget": {"total_price": 660},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["cheap_vague_lodging"]


def test_constraint_filter_recognizes_lodging_node_type_for_soft_budget():
    state = {
        "constraints": {
            "raw_text": "\u60f3\u627e\u4e2a\u4fbf\u5b9c\u70b9\u7684\u5730\u65b9\u8fc7\u591c",
            "budget": 300,
            "budget_type": "total",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "lodging_type_node",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 2,
                "planning_horizon": "overnight",
                "nodes": [
                    {"poi_id": "hotel_1", "type": "lodging", "itinerary_role": "lodging", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                ],
                "route": {
                    "total_distance_km": 4,
                    "total_travel_time_min": 25,
                    "legs": [{"distance_km": 4}],
                },
                "budget": {"total_price": 720},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["lodging_type_node"]


def test_constraint_filter_does_not_treat_year_or_clock_as_budget_signal():
    state = {
        "constraints": {
            "raw_text": "2025\u4e0a\u6d77\u513f\u7ae5\u620f\u5267\u8282\uff0c\u4e0b\u53485\u70b9\u540e\u627e\u9910\u5385",
            "budget": 300,
            "budget_type": "total",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "year_and_clock_not_budget",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 2,
                "planning_horizon": "overnight",
                "nodes": [
                    {"poi_id": "act_1", "type": "activity", "itinerary_role": "family_activity", "available": True},
                    {"poi_id": "hotel_1", "type": "lodging", "itinerary_role": "lodging", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                ],
                "route": {
                    "total_distance_km": 5,
                    "total_travel_time_min": 35,
                    "legs": [{"distance_km": 2}, {"distance_km": 3}],
                },
                "budget": {"total_price": 720},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 900,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["year_and_clock_not_budget"]


def test_constraint_filter_allows_unpriced_multi_node_without_explicit_budget():
    state = {
        "constraints": {
            "raw_text": "\u6c49\u670d\u62cd\u7167\uff0c\u8336\u827a\u4f11\u606f\uff0c\u7136\u540e\u5403\u665a\u996d",
            "budget": 500,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 720],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "cultural_tea_dinner",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "full_day",
                "nodes": [
                    {"poi_id": "photo_1", "type": "activity", "itinerary_role": "cultural_photo", "available": True},
                    {"poi_id": "tea_1", "type": "restaurant", "itinerary_role": "tea_house", "available": True},
                    {"poi_id": "spa_1", "type": "activity", "itinerary_role": "wellness_massage", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_dinner", "available": True},
                ],
                "route": {
                    "total_distance_km": 5,
                    "total_travel_time_min": 35,
                    "legs": [{"distance_km": 1}, {"distance_km": 2}, {"distance_km": 2}],
                },
                "budget": {"total_price": 956},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 630,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["cultural_tea_dinner"]


def test_constraint_filter_expands_per_person_budget_for_large_group():
    state = {
        "constraints": {
            "raw_text": "15\u4e2a\u540c\u4e8b\u4eba\u5747300\u9884\u7b97\u6253\u684c\u6e38\u5403\u996d\u5531K",
            "budget": 300,
            "budget_type": "total",
            "people_count": 15,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 480],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "large_group_per_person_budget",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {"poi_id": "board_1", "type": "activity", "itinerary_role": "board_game_escape", "available": True},
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_dinner", "available": True},
                    {"poi_id": "ktv_1", "type": "activity", "itinerary_role": "karaoke", "available": True},
                ],
                "route": {
                    "total_distance_km": 5,
                    "total_travel_time_min": 35,
                    "legs": [{"distance_km": 2}, {"distance_km": 3}],
                },
                "budget": {"total_price": 1800},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 390,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["large_group_per_person_budget"]


def test_candidate_generator_attaches_requirement_contract(monkeypatch):
    _clear_env(monkeypatch)
    result = candidate_generator_node(
        {
            "user_input": "\u4eca\u5929\u4e0b\u5348\u60f3\u548c\u8001\u5a46\u5b69\u5b50\u51fa\u53bb\u73a9\uff0c\u5b69\u5b505\u5c81\u3002",
            "scene_type": "family",
            "constraints": {"child_age": 5, "people_count": 3},
            "user_profile": {},
            "scenario_activities": [],
            "execution_log": [],
        }
    )

    assert result["b_requirement_contract"]["hard_requirements"] == ["child_friendly_activity"]
    assert "b_requirement_contract" in result["constraints"]
