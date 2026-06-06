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


def test_requirement_compiler_does_not_promote_rejected_old_night_snack_preference(monkeypatch):
    _clear_env(monkeypatch)
    _, contract, _ = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": (
                "周日想上午买伴手礼，中午简单吃，下午再有一个不累的亲子点，"
                "16:30前回到家附近。爸爸以前爱密室和重口味夜宵，这次别按那个来。"
            ),
            "scene_type": "family",
            "constraints": {"people_count": 3},
        },
        constraints={"people_count": 3},
        allow_llm=False,
    )

    assert "late_night_open" not in contract["hard_requirements"]
    assert "child_friendly_activity" in contract["hard_requirements"]


def test_requirement_compiler_family_scene_without_child_does_not_force_child_guard(monkeypatch):
    _clear_env(monkeypatch)
    _, contract, _ = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": (
                "周六陪妈妈从医院复诊出来，吃个低盐清淡午饭，"
                "妈妈膝盖不好不能久走，预算总共800以内。"
            ),
            "scene_type": "family",
            "constraints": {
                "raw_text": (
                    "周六陪妈妈从医院复诊出来，吃个低盐清淡午饭，"
                    "妈妈膝盖不好不能久走，预算总共800以内。"
                ),
                "scene": "family",
                "people_count": 2,
            },
        },
        constraints={
            "raw_text": (
                "周六陪妈妈从医院复诊出来，吃个低盐清淡午饭，"
                "妈妈膝盖不好不能久走，预算总共800以内。"
            ),
            "scene": "family",
            "people_count": 2,
        },
        allow_llm=False,
    )

    assert "elder_friendly" in contract["hard_requirements"]
    assert "child_friendly_activity" not in contract["hard_requirements"]


def test_requirement_compiler_respects_current_turn_no_child_override(monkeypatch):
    _clear_env(monkeypatch)
    _, contract, _ = b_requirement_compiler.apply_b_requirement_contract(
        {
            "user_input": "今晚和老婆过纪念日，孩子这次不带，别按亲子来排。",
            "scene_type": "couple",
            "constraints": {
                "raw_text": "今晚和老婆过纪念日，孩子这次不带，别按亲子来排。",
                "scene": "couple",
                "avoid": ["亲子", "儿童友好"],
            },
        },
        constraints={
            "raw_text": "今晚和老婆过纪念日，孩子这次不带，别按亲子来排。",
            "scene": "couple",
            "avoid": ["亲子", "儿童友好"],
        },
        allow_llm=False,
    )

    assert "child_friendly_activity" not in contract["hard_requirements"]


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


def test_constraint_filter_summary_does_not_claim_zero_candidates_satisfy_constraints():
    state = {
        "constraints": {
            "budget": 600,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["_summary"] == "未生成候选方案，因此没有可执行方案满足硬约束"
    assert "全部满足硬约束" not in result["filter_reasons"]["_summary"]


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


def test_constraint_filter_rejects_barbecue_even_with_high_protein_for_low_calorie_context():
    state = {
        "constraints": {
            "raw_text": "\u4eca\u5929\u548c\u8001\u5a46\u5403\u665a\u9910\uff0c\u5979\u6700\u8fd1\u5728\u51cf\u8102",
            "scene": "couple",
            "mom_diet": "low_calorie",
            "budget": 800,
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
                "high_protein_bbq_plan",
                {"name": "\u6c5f\u666f\u6563\u6b65", "tags": ["\u6c5f\u666f"]},
                {
                    "name": "Latina\u5df4\u897f\u725b\u6392\u9986",
                    "tags": ["\u70e4\u8089", "\u725b\u6392"],
                    "health_tags": ["high_protein"],
                    "restaurant_category": "\u70e4\u8089",
                },
            )
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["high_protein_bbq_plan"] == "\u4e0d\u7b26\u5408\u4f4e\u5361\u6216\u8f7b\u98df\u9700\u6c42"


def test_constraint_filter_accepts_chinese_low_calorie_health_signals():
    plan = _plan(
        "cn_light_food_plan",
        {"name": "\u827a\u672f\u5c55", "tags": ["\u770b\u5c55"]},
        {
            "name": "\u5065\u5eb7\u8f7b\u98df\u9910\u5385",
            "tags": ["\u8f7b\u98df", "\u5065\u5eb7"],
            "health_tags": ["\u4f4e\u5361", "\u5c11\u6cb9", "\u852c\u83dc\u4e30\u5bcc"],
            "restaurant_category": "\u8f7b\u98df",
        },
    )
    state = {
        "constraints": {
            "raw_text": "\u665a\u4e0a\u60f3\u5403\u70b9\u4e0d\u6cb9\u817b\u7684\u5065\u5eb7\u9910",
            "mom_diet": "low_calorie",
            "budget": 300,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [plan],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["cn_light_food_plan"]


def test_constraint_filter_softens_distance_without_concrete_origin_coordinates():
    plan = _plan(
        "nearby_without_geo_origin",
        {"name": "\u4eb2\u5b50\u624b\u4f5c", "tags": ["\u4eb2\u5b50"]},
        {"name": "\u8f7b\u98df\u9910\u5385", "tags": ["light_food"], "health_tags": ["low_calorie"]},
    )
    plan["route"] = {"total_distance_km": 12, "total_travel_time_min": 35}
    state = {
        "constraints": {
            "raw_text": "\u522b\u79bb\u5bb6\u592a\u8fdc",
            "origin": "home",
            "budget": 800,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [plan],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["nearby_without_geo_origin"]


def test_constraint_filter_keeps_distance_hard_with_concrete_origin_coordinates():
    plan = _plan(
        "far_with_geo_origin",
        {"name": "\u4eb2\u5b50\u624b\u4f5c", "tags": ["\u4eb2\u5b50"]},
        {"name": "\u8f7b\u98df\u9910\u5385", "tags": ["light_food"], "health_tags": ["low_calorie"]},
    )
    plan["route"] = {"total_distance_km": 12, "total_travel_time_min": 35}
    state = {
        "constraints": {
            "raw_text": "\u522b\u79bb\u5bb6\u592a\u8fdc",
            "origin": "home",
            "origin_coordinates": "121.49,31.24",
            "budget": 800,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [3, 5],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [plan],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["far_with_geo_origin"] == "\u8ddd\u79bb\u8d85\u8fc7\u7528\u6237\u53ef\u63a5\u53d7\u8303\u56f4"


def test_constraint_filter_softens_walking_leg_distance_without_explicit_km():
    state = {
        "constraints": {
            "raw_text": "\u4e00\u6574\u5929\u90fd\u5728\u4e00\u4e2a\u533a\u57df\u96c6\u4e2d\uff0c\u8d70\u8def\u5c31\u80fd\u5230",
            "route_mode": "walking",
            "budget": 800,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 1,
            "max_queue_time_min": 30,
            "duration_range": [6, 10],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "walkable_cluster",
                "planner_mode": "multi_node_itinerary",
                "planning_horizon": "full_day",
                "nodes": [
                    {"type": "activity", "name": "\u6c49\u670d\u9986", "available": True},
                    {"type": "restaurant", "name": "\u9910\u5385", "available": True},
                    {"type": "restaurant", "name": "\u8336\u9986", "available": True},
                ],
                "route": {
                    "total_distance_km": 4.2,
                    "total_travel_time_min": 55,
                    "legs": [{"distance_km": 2.1}, {"distance_km": 2.1}],
                },
                "budget": {"total_price": 520},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 480,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["walkable_cluster"]


def test_constraint_filter_respects_explicit_walking_distance_number():
    state = {
        "constraints": {
            "raw_text": "\u6bcf\u6bb5\u6700\u597d1\u516c\u91cc\u5185",
            "route_mode": "walking",
            "budget": 800,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 1,
            "max_queue_time_min": 30,
            "duration_range": [6, 10],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "explicit_too_far_walk",
                "planner_mode": "multi_node_itinerary",
                "planning_horizon": "full_day",
                "nodes": [
                    {"type": "activity", "name": "\u6c49\u670d\u9986", "available": True},
                    {"type": "restaurant", "name": "\u9910\u5385", "available": True},
                    {"type": "restaurant", "name": "\u8336\u9986", "available": True},
                ],
                "route": {
                    "total_distance_km": 4.2,
                    "total_travel_time_min": 55,
                    "legs": [{"distance_km": 2.1}, {"distance_km": 2.1}],
                },
                "budget": {"total_price": 520},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 480,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["explicit_too_far_walk"] == "\u5355\u6bb5\u8ddd\u79bb\u8d85\u8fc7\u7528\u6237\u53ef\u63a5\u53d7\u8303\u56f4"


def test_constraint_filter_softens_text_anchor_multinode_distance_without_coordinates():
    state = {
        "constraints": {
            "raw_text": "\u6211\u5728\u5916\u6ee9\u9644\u8fd1\uff0c\u665a\u4e0a\u60f3\u5403\u5065\u5eb7\u9910\uff0c\u518d\u627e\u4fbf\u5229\u5e97\u548c\u5496\u5561\u5385",
            "route_mode": "walking",
            "origin": "\u5916\u6ee9\u9644\u8fd1",
            "budget": 800,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 2,
            "max_queue_time_min": 30,
            "duration_range": [1, 6],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "text_anchor_cluster",
                "planner_mode": "multi_node_itinerary",
                "nodes": [
                    {"type": "restaurant", "name": "\u5065\u5eb7\u9910", "available": True},
                    {"type": "shopping", "name": "\u4fbf\u5229\u5e97", "available": True},
                    {"type": "restaurant", "name": "\u5496\u5561\u5385", "available": True},
                ],
                "route": {
                    "total_distance_km": 7.0,
                    "total_travel_time_min": 55,
                    "legs": [{"distance_km": 3.0}, {"distance_km": 3.4}, {"distance_km": 0.6}],
                },
                "budget": {"total_price": 220},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 190,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["text_anchor_cluster"]


def test_constraint_filter_softens_driving_distance_with_text_only_anchor():
    state = {
        "constraints": {
            "raw_text": "\u6d3b\u52a8\u4e3e\u529e\u5730\u9644\u8fd1\u627e\u9910\u5385\u3001\u5496\u5561\u5385\u3001\u4fbf\u5229\u5e97\u548c\u505c\u8f66\u573a",
            "route_mode": "driving",
            "origin": "\u6d3b\u52a8\u4e3e\u529e\u5730",
            "budget": 800,
            "budget_type": "total",
            "people_count": 1,
            "max_distance_km": 2,
            "max_queue_time_min": 30,
            "duration_range": [1, 8],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "text_anchor_driving_cluster",
                "planner_mode": "multi_node_itinerary",
                "nodes": [
                    {"type": "restaurant", "name": "\u9910\u5385", "available": True},
                    {"type": "restaurant", "name": "\u5496\u5561\u5385", "available": True},
                    {"type": "shopping", "name": "\u4fbf\u5229\u5e97", "available": True},
                    {"type": "transport_service", "name": "\u505c\u8f66\u573a", "available": True},
                ],
                "route": {
                    "total_distance_km": 16.0,
                    "total_travel_time_min": 65,
                    "legs": [
                        {"distance_km": 3.0},
                        {"distance_km": 7.0},
                        {"distance_km": 4.0},
                        {"distance_km": 2.0},
                    ],
                },
                "budget": {"total_price": 260},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 220,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["text_anchor_driving_cluster"]


def test_constraint_filter_treats_small_baozi_budget_as_item_scoped():
    state = {
        "constraints": {
            "raw_text": "\u670910\u5757\u94b1\u4ee5\u5185\u80fd\u5403\u9971\u7684\u5305\u5b50\u5e97\uff0c\u8fd8\u60f3\u627e\u4e2a\u4fbf\u5229\u5e97",
            "budget": 10,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [1, 6],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "baozi_plus_shop",
                "planner_mode": "multi_node_itinerary",
                "nodes": [
                    {
                        "type": "restaurant",
                        "name": "\u5305\u5b50\u5e97",
                        "itinerary_role": "restaurant_breakfast",
                        "price": 10,
                        "available": True,
                    },
                    {
                        "type": "shopping",
                        "name": "\u4fbf\u5229\u5e97",
                        "itinerary_role": "convenience_store",
                        "price": 80,
                        "available": True,
                    },
                ],
                "route": {
                    "total_distance_km": 2.0,
                    "total_travel_time_min": 25,
                    "legs": [{"distance_km": 1.0}, {"distance_km": 1.0}],
                },
                "budget": {"total_price": 90},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 90,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["baozi_plus_shop"]


def test_constraint_filter_keeps_total_budget_hard_when_small_budget_is_global():
    state = {
        "constraints": {
            "raw_text": "\u603b\u9884\u7b9710\u5143\u4ee5\u5185\u5b89\u6392\u65e9\u9910\u548c\u4fbf\u5229\u5e97",
            "budget": 10,
            "budget_type": "total",
            "people_count": 1,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [1, 6],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "global_ten_yuan_plan",
                "planner_mode": "multi_node_itinerary",
                "nodes": [
                    {"type": "restaurant", "name": "\u65e9\u9910", "price": 10, "available": True},
                    {"type": "shopping", "name": "\u4fbf\u5229\u5e97", "price": 80, "available": True},
                ],
                "route": {
                    "total_distance_km": 2.0,
                    "total_travel_time_min": 25,
                    "legs": [{"distance_km": 1.0}, {"distance_km": 1.0}],
                },
                "budget": {"total_price": 90},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 90,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["global_ten_yuan_plan"] == "\u9884\u7b97\u8d85\u51fa\u53ef\u63a5\u53d7\u4e0a\u9650"


def test_constraint_filter_ignores_synthetic_first_leg_without_origin_for_multinode():
    state = {
        "constraints": {
            "raw_text": "\u4e00\u6574\u5929\u90fd\u5728\u4e00\u4e2a\u533a\u57df\u96c6\u4e2d\uff0c\u8d70\u8def\u5c31\u80fd\u5230",
            "route_mode": "walking",
            "origin": "home",
            "budget": 800,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 1,
            "max_queue_time_min": 30,
            "duration_range": [6, 10],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "cluster_with_synthetic_start",
                "planner_mode": "multi_node_itinerary",
                "planning_horizon": "full_day",
                "nodes": [
                    {"type": "activity", "name": "\u6c49\u670d\u9986", "available": True},
                    {"type": "restaurant", "name": "\u9910\u5385", "available": True},
                    {"type": "restaurant", "name": "\u8336\u9986", "available": True},
                ],
                "route": {
                    "total_distance_km": 5.4,
                    "total_travel_time_min": 60,
                    "legs": [{"distance_km": 3.0}, {"distance_km": 0.8}, {"distance_km": 1.6}],
                },
                "budget": {"total_price": 520},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 480,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [item["plan_id"] for item in result["filtered_candidates"]] == ["cluster_with_synthetic_start"]


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


def test_constraint_filter_allows_bakery_as_cafe_rest_stop():
    state = {
        "constraints": {
            "budget": 1000,
            "max_distance_km": 30,
            "max_queue_time_min": 30,
            "duration_range": [180, 1000],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "massage_bakery_dinner_shopping_parking",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "full_day",
                "nodes": [
                    {
                        "poi_id": "spa_1",
                        "type": "activity",
                        "itinerary_role": "wellness_massage",
                        "name": "麦悠悠·SPA·推拿",
                        "available": True,
                    },
                    {
                        "poi_id": "cafe_1",
                        "type": "restaurant",
                        "itinerary_role": "cafe",
                        "name": "HOTCRUSH趁热集合·现烤面包",
                        "primary_category": "糕饼店",
                        "available": True,
                        "dine_in_available": True,
                    },
                    {
                        "poi_id": "res_1",
                        "type": "restaurant",
                        "itinerary_role": "restaurant_specific",
                        "name": "烧肉二十九号",
                        "available": True,
                        "dine_in_available": True,
                    },
                    {
                        "poi_id": "shop_1",
                        "type": "shopping",
                        "itinerary_role": "convenience_store",
                        "name": "全家便利店",
                        "available": True,
                    },
                    {
                        "poi_id": "parking_1",
                        "type": "transport_service",
                        "itinerary_role": "parking",
                        "name": "商圈停车指引",
                        "parking_proxy": True,
                        "available": True,
                    },
                ],
                "route": {"total_distance_km": 8, "total_travel_time_min": 45},
                "budget": {"total_price": 500},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 360,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == [
        "massage_bakery_dinner_shopping_parking"
    ]


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


def test_constraint_filter_does_not_apply_default_lower_duration_to_short_recommendation():
    state = {
        "constraints": {
            "raw_text": "\u9759\u5b89\u5bfa\u9644\u8fd1\u627e\u4e2aSPA\u548c\u9910\u5385",
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
            _plan(
                "short_recommendation",
                {"name": "SPA\u63a8\u62ff", "tags": ["SPA", "\u6309\u6469"]},
                {"name": "\u672c\u5e2e\u9910\u5385", "tags": ["\u9910\u5385"]},
            )
        ],
        "execution_log": [],
    }
    state["candidates"][0]["estimated_duration_min"] = 140

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["short_recommendation"]


def test_constraint_filter_applies_lower_duration_when_user_explicitly_requests_hours():
    state = {
        "constraints": {
            "raw_text": "\u60f3\u73a93\u4e2a\u5c0f\u65f6\u5de6\u53f3",
            "budget": 1200,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 240],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            _plan(
                "too_short_for_explicit_duration",
                {"name": "\u4f11\u95f2\u6d3b\u52a8", "tags": ["\u5ba4\u5185"]},
                {"name": "\u9910\u5385", "tags": ["\u9910\u5385"]},
            )
        ],
        "execution_log": [],
    }
    state["candidates"][0]["estimated_duration_min"] = 120

    result = constraint_filter_node(state)

    assert result["filtered_candidates"] == []
    assert result["filter_reasons"]["too_short_for_explicit_duration"] == "\u65f6\u957f\u4e0d\u6ee1\u8db3\u7528\u6237\u7684\u65f6\u95f4\u8303\u56f4"


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


def test_constraint_filter_does_not_treat_venue_hours_as_total_duration():
    state = {
        "constraints": {
            "raw_text": "想吃日料，再做个按摩，最后找个24小时健身房锻炼1小时",
            "budget": 1200,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [60, 240],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "venue_hours_are_not_total_duration",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {"poi_id": "res_1", "type": "restaurant", "itinerary_role": "restaurant_specific", "available": True},
                    {"poi_id": "spa_1", "type": "activity", "itinerary_role": "wellness_massage", "available": True},
                    {"poi_id": "gym_1", "type": "activity", "itinerary_role": "fitness", "available": True},
                ],
                "route": {
                    "total_distance_km": 4,
                    "total_travel_time_min": 30,
                    "legs": [{"distance_km": 2}, {"distance_km": 2}],
                },
                "budget": {"total_price": 760},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 390,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["venue_hours_are_not_total_duration"]


def test_constraint_filter_allows_pet_specific_nodes_for_pet_friendly_contract():
    state = {
        "constraints": {
            "raw_text": "带金毛做宠物美容，之后去宠物友好咖啡馆，再去宠物医院体检",
            "budget": 1200,
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 420],
            "b_requirement_contract": {
                "hard_requirements": ["pet_friendly"],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "pet_specific_nodes",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {"poi_id": "pet_grooming_1", "type": "pet_service", "itinerary_role": "pet_grooming", "available": True},
                    {"poi_id": "pet_cafe_1", "type": "restaurant", "itinerary_role": "pet_cafe", "name": "宠物友好咖啡", "available": True},
                    {"poi_id": "pet_hospital_1", "type": "pet_service", "itinerary_role": "pet_hospital", "available": True},
                ],
                "route": {
                    "total_distance_km": 4,
                    "total_travel_time_min": 30,
                    "legs": [{"distance_km": 2}, {"distance_km": 2}],
                },
                "budget": {"total_price": 500},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 300,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["pet_specific_nodes"]


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


def test_constraint_filter_allows_cultural_photo_museum_and_culture_street_nodes():
    state = {
        "constraints": {
            "raw_text": "\u4e0a\u6d77\u73a9\u4e00\u5929 \u62cd\u7167 \u804a\u5929 \u4e0d\u5403\u8fa3 \u4eba\u5747200 \u53ef\u80fd\u4e0b\u96e8",
            "budget": 200,
            "budget_type": "per_person",
            "people_count": 4,
            "max_distance_km": 16,
            "max_queue_time_min": 30,
            "duration_range": [180, 720],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "museum_lunch_tianzifang",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "full_day",
                "nodes": [
                    {
                        "poi_id": "gaode_act_B0FFI2885X",
                        "name": "\u4e0a\u6d77\u5e02\u5386\u53f2\u535a\u7269\u9986",
                        "type": "activity",
                        "category": "\u535a\u7269\u9986",
                        "tags": ["museum", "local_culture", "indoor"],
                        "itinerary_role": "cultural_photo",
                        "available": True,
                    },
                    {
                        "poi_id": "gaode_res_B001513877",
                        "name": "\u745e\u798f\u56ed(\u8302\u540d\u5357\u8def\u5e97)",
                        "type": "restaurant",
                        "restaurant_category": "\u672c\u5e2e\u83dc",
                        "itinerary_role": "restaurant_lunch",
                        "available": True,
                        "dine_in_available": True,
                    },
                    {
                        "poi_id": "gaode_act_B00155HO6Y",
                        "name": "\u4e0a\u6d77\u7530\u5b50\u574a",
                        "type": "activity",
                        "category": "local_market",
                        "tags": ["local_culture", "citywalk", "\u6587\u5316\u8857\u533a"],
                        "itinerary_role": "cultural_photo",
                        "available": True,
                    },
                ],
                "route": {
                    "total_distance_km": 6,
                    "total_travel_time_min": 45,
                    "legs": [{"distance_km": 2}, {"distance_km": 2}, {"distance_km": 2}],
                },
                "budget": {"total_price": 290},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 510,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["museum_lunch_tianzifang"]


def test_constraint_filter_rejects_sports_activity_for_relaxed_leisure_request():
    state = {
        "constraints": {
            "raw_text": "\u5403\u5b8c\u70e4\u8089\u627e\u4e2a\u8f7b\u677e\u6d3b\u52a8\uff0c\u4e0d\u8981\u592a\u7d2f",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 420],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            _plan(
                "sports_like",
                {
                    "name": "PURE Yoga & Fitness",
                    "category": "\u5065\u8eab\u4e2d\u5fc3",
                    "tags": ["\u745c\u4f3d", "\u5065\u8eab", "\u8fd0\u52a8\u4f53\u9a8c"],
                },
                {"name": "\u65e5\u5f0f\u70e7\u8089", "tags": ["\u70e4\u8089"]},
            ),
            _plan(
                "relaxed_chat",
                {
                    "name": "\u684c\u6e38\u5c0f\u9986",
                    "category": "\u684c\u6e38",
                    "tags": ["\u5ba4\u5185", "\u804a\u5929", "\u8f7b\u677e"],
                },
                {"name": "\u65e5\u5f0f\u70e7\u8089", "tags": ["\u70e4\u8089"]},
            ),
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["relaxed_chat"]


def test_constraint_filter_allows_sports_activity_when_user_explicitly_asks_for_sports():
    state = {
        "constraints": {
            "raw_text": "\u5403\u5b8c\u706b\u9505\u53bb\u8fd0\u52a8",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 420],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            _plan(
                "sports_explicit",
                {
                    "name": "PURE Yoga & Fitness",
                    "category": "\u5065\u8eab\u4e2d\u5fc3",
                    "tags": ["\u745c\u4f3d", "\u5065\u8eab", "\u8fd0\u52a8\u4f53\u9a8c"],
                },
                {"name": "\u706b\u9505\u5e97", "tags": ["\u706b\u9505"]},
            ),
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["sports_explicit"]


def test_constraint_filter_rejects_mismatched_multinode_role_fillers():
    state = {
        "constraints": {
            "raw_text": "\u627e\u4e2a\u5496\u5561\u5385\u5750\u5750\uff0c\u518d\u53bb\u4fbf\u5229\u5e97\u4e70\u96f6\u98df",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [60, 240],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "wrong_cafe",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {
                        "poi_id": "res_1",
                        "type": "restaurant",
                        "itinerary_role": "cafe",
                        "itinerary_label": "\u5496\u5561/\u4e0b\u5348\u8336",
                        "_itinerary_intent": {"role": "cafe", "label": "\u5496\u5561/\u4e0b\u5348\u8336"},
                        "name": "\u732a\u6392\u996d\u9910\u5385",
                        "category": "\u5496\u5561/\u4e0b\u5348\u8336",
                        "restaurant_category": "\u65e5\u6599",
                        "tags": ["\u65e5\u6599", "\u6b63\u9910", "\u5496\u5561/\u4e0b\u5348\u8336", "cafe"],
                        "available": True,
                    },
                    {
                        "poi_id": "shop_1",
                        "type": "restaurant",
                        "itinerary_role": "convenience_store",
                        "itinerary_label": "\u4fbf\u5229\u5e97/\u65e5\u7528\u54c1",
                        "_itinerary_intent": {"role": "convenience_store", "label": "\u4fbf\u5229\u5e97/\u65e5\u7528\u54c1"},
                        "name": "\u70e4\u8089\u5e97",
                        "category": "\u4fbf\u5229\u5e97/\u65e5\u7528\u54c1",
                        "tags": ["\u70e4\u8089", "\u4fbf\u5229\u5e97/\u65e5\u7528\u54c1"],
                        "available": True,
                    },
                ],
                "route": {"total_distance_km": 2, "total_travel_time_min": 20, "legs": [{"distance_km": 2}]},
                "budget": {"total_price": 120},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 120,
            },
            {
                "plan_id": "right_roles",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 1,
                "planning_horizon": "half_day",
                "nodes": [
                    {
                        "poi_id": "cafe_1",
                        "type": "restaurant",
                        "itinerary_role": "cafe",
                        "name": "\u5b89\u9759\u5496\u5561\u9986",
                        "tags": ["\u5496\u5561", "\u4e0b\u5348\u8336"],
                        "available": True,
                    },
                    {
                        "poi_id": "shop_2",
                        "type": "shopping",
                        "itinerary_role": "convenience_store",
                        "name": "\u5168\u5bb6\u4fbf\u5229\u5e97",
                        "tags": ["\u4fbf\u5229\u5e97", "\u96f6\u98df", "\u996e\u6599"],
                        "available": True,
                    },
                ],
                "route": {"total_distance_km": 2, "total_travel_time_min": 20, "legs": [{"distance_km": 2}]},
                "budget": {"total_price": 120},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 120,
            },
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["right_roles"]


def test_constraint_filter_does_not_apply_child_activity_requirement_without_activity_node():
    state = {
        "constraints": {
            "raw_text": "\u53c2\u52a0\u513f\u7ae5\u620f\u5267\u8282\u540e\u4f4f\u4e00\u665a\u518d\u5403\u996d",
            "max_distance_km": 8,
            "max_queue_time_min": 30,
            "duration_range": [180, 900],
            "b_requirement_contract": {
                "hard_requirements": ["child_friendly_activity"],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "lodging_restaurant_only",
                "planner_mode": "multi_node_itinerary",
                "planning_horizon": "overnight",
                "planning_days": 2,
                "nodes": [
                    {
                        "poi_id": "hotel_1",
                        "type": "hotel",
                        "itinerary_role": "lodging",
                        "name": "\u9759\u5b89\u96c5\u81f4\u6c11\u5bbf",
                        "tags": ["\u4f4f\u5bbf", "\u6c11\u5bbf"],
                        "available": True,
                    },
                    {
                        "poi_id": "res_1",
                        "type": "restaurant",
                        "itinerary_role": "restaurant_specific",
                        "name": "\u9759\u5b89\u96c5\u7d20\u9601",
                        "tags": ["\u9910\u5385"],
                        "available": True,
                    },
                ],
                "route": {"total_distance_km": 2, "total_travel_time_min": 20, "legs": [{"distance_km": 2}]},
                "budget": {"total_price": 500},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 850,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["lodging_restaurant_only"]


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


def test_constraint_filter_treats_multi_node_item_budgets_as_scoped():
    state = {
        "constraints": {
            "raw_text": (
                "周六晚上7点看话剧，之后附近吃夜宵（人均150元左右）。"
                "周日上午去文艺咖啡馆（人均50元），下午2点看相声，"
                "然后买些上海特产伴手礼（预算200元）。"
            ),
            "budget": 150,
            "budget_type": "per_person",
            "people_count": 1,
            "max_distance_km": 15,
            "max_queue_time_min": 30,
            "duration_range": [0, 1200],
            "b_requirement_contract": {
                "hard_requirements": [],
                "forbidden_restaurant_groups": [],
            },
        },
        "candidates": [
            {
                "plan_id": "scoped_item_budget_multinode",
                "planner_mode": "multi_node_itinerary",
                "planning_days": 2,
                "planning_horizon": "two_day",
                "nodes": [
                    {"poi_id": "theatre_1", "name": "青年话剧剧院", "type": "activity", "itinerary_role": "theatre_performance", "price": 168, "available": True},
                    {"poi_id": "snack_1", "name": "夜宵烧烤餐厅", "type": "restaurant", "itinerary_role": "restaurant_specific", "price": 96, "available": True},
                    {"poi_id": "cafe_1", "name": "文艺咖啡馆", "type": "restaurant", "itinerary_role": "cafe", "price": 49, "available": True},
                    {"poi_id": "talk_1", "name": "相声演出剧场", "type": "activity", "itinerary_role": "talk_show", "price": 160, "available": True},
                    {"poi_id": "gift_1", "name": "上海特产伴手礼", "type": "shopping", "itinerary_role": "souvenir_shopping", "price": 80, "available": True},
                ],
                "route": {
                    "total_distance_km": 8,
                    "total_travel_time_min": 60,
                    "legs": [{"distance_km": 2}, {"distance_km": 2}, {"distance_km": 2}, {"distance_km": 2}],
                },
                "budget": {"total_price": 553},
                "availability": {"all_available": True, "max_queue_time_min": 10},
                "estimated_duration_min": 520,
            }
        ],
        "execution_log": [],
    }

    result = constraint_filter_node(state)

    assert [plan["plan_id"] for plan in result["filtered_candidates"]] == ["scoped_item_budget_multinode"]


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


def _single_role_plan(plan_id: str, node: dict) -> dict:
    return {
        "plan_id": plan_id,
        "planner_mode": "multi_node_itinerary",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "nodes": [node],
        "route": {"total_distance_km": 0.5, "total_travel_time_min": 8, "legs": [{"distance_km": 0.5}]},
        "budget": {"total_price": 80},
        "availability": {"all_available": True, "max_queue_time_min": 5},
        "estimated_duration_min": 60,
    }


def _filter_plan_ids(candidates: list[dict]) -> list[str]:
    result = constraint_filter_node(
        {
            "constraints": {
                "raw_text": "多业态本地生活链路",
                "max_distance_km": 8,
                "max_queue_time_min": 30,
                "duration_range": [0, 720],
            },
            "candidates": candidates,
            "execution_log": [],
        }
    )
    return [plan["plan_id"] for plan in result["filtered_candidates"]]


def test_constraint_filter_rejects_karaoke_role_impostors():
    candidates = [
        _single_role_plan(
            "real_ktv",
            {
                "poi_id": "ktv_1",
                "type": "activity",
                "itinerary_role": "karaoke",
                "name": "魅KTVPlus·AI辅唱",
                "tags": ["KTV", "唱歌"],
                "available": True,
            },
        ),
        _single_role_plan(
            "foot_massage_k_song",
            {
                "poi_id": "spa_ktv_1",
                "type": "activity",
                "itinerary_role": "karaoke",
                "name": "澜庭k歌沐足",
                "tags": ["足疗", "按摩", "k歌"],
                "available": True,
            },
        ),
        _single_role_plan(
            "exhibition_fake_karaoke",
            {
                "poi_id": "exhibition_1",
                "type": "activity",
                "itinerary_role": "karaoke",
                "name": "巴黎1874·印象派之夜",
                "tags": ["展览", "艺术"],
                "available": True,
            },
        ),
    ]

    assert _filter_plan_ids(candidates) == ["real_ktv"]


def test_constraint_filter_rejects_retail_role_impostors():
    candidates = [
        _single_role_plan(
            "real_souvenir",
            {
                "poi_id": "souvenir_1",
                "type": "shopping",
                "itinerary_role": "souvenir_shopping",
                "name": "宁波土特产商行",
                "tags": ["土特产", "伴手礼"],
                "available": True,
            },
        ),
        _single_role_plan(
            "coffee_fake_souvenir",
            {
                "poi_id": "coffee_1",
                "type": "shopping",
                "itinerary_role": "souvenir_shopping",
                "name": "Peet's皮爷咖啡",
                "tags": ["咖啡", "下午茶", "伴手礼"],
                "available": True,
            },
        ),
        _single_role_plan(
            "real_flower",
            {
                "poi_id": "flower_1",
                "type": "shopping",
                "itinerary_role": "flower_shop",
                "name": "天天鲜花",
                "tags": ["鲜花", "花店"],
                "available": True,
            },
        ),
        _single_role_plan(
            "miniso_fake_flower",
            {
                "poi_id": "miniso_1",
                "type": "shopping",
                "itinerary_role": "flower_shop",
                "name": "miniso land(CP静安店)",
                "tags": ["鲜花", "礼品"],
                "available": True,
            },
        ),
    ]

    assert _filter_plan_ids(candidates) == ["real_souvenir", "real_flower"]


def test_constraint_filter_rejects_cafe_or_bar_as_full_meal_roles():
    candidates = [
        _single_role_plan(
            "real_lunch",
            {
                "poi_id": "lunch_1",
                "type": "restaurant",
                "itinerary_role": "restaurant_lunch",
                "name": "双合园·海鲜水饺青岛菜",
                "tags": ["正餐", "青岛菜"],
                "available": True,
            },
        ),
        _single_role_plan(
            "cafe_fake_lunch",
            {
                "poi_id": "cafe_1",
                "type": "restaurant",
                "itinerary_role": "restaurant_lunch",
                "name": "1691cafebar",
                "tags": ["咖啡", "bar", "下午茶"],
                "available": True,
            },
        ),
    ]

    assert _filter_plan_ids(candidates) == ["real_lunch"]
