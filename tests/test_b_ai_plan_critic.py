from __future__ import annotations

import json

from src.nodes import b_ai_plan_critic
from src.nodes import plan_optimizer as plan_optimizer_module


def _clear_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "WF_B_AI_SEMANTIC_HINTS_ENABLED",
        "WF_B_AI_PLAN_CRITIC_ENABLED",
        "WF_B_AI_PLAN_CRITIC_TOP_K",
        "WF_B_AI_PLAN_CRITIC_MAX_SCORE_DELTA",
        "WF_B_AI_PLAN_CRITIC_MAX_RISK_DELTA",
        "WF_B_AI_PLAN_CRITIC_ALWAYS",
        "WF_B_AI_PLAN_CRITIC_SKIP_MIN_SCORE_GAP",
        "WF_B_AI_PLAN_CRITIC_SKIP_MAX_RISK",
        "WF_B_AI_PLAN_CRITIC_SKIP_MAX_RISK_FACTORS",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _candidate(plan_id: str, score: float, *, category: str = "light_food") -> dict:
    return {
        "plan": {
            "plan_id": plan_id,
            "nodes": [
                {
                    "poi_id": f"act_{plan_id}",
                    "type": "activity",
                    "name": f"Activity {plan_id}",
                    "category": "indoor_activity",
                    "tags": ["indoor", "low_intensity"],
                    "rating": 4.6,
                    "queue_time_min": 8,
                },
                {
                    "poi_id": f"res_{plan_id}",
                    "type": "restaurant",
                    "name": f"Restaurant {plan_id}",
                    "restaurant_category": category,
                    "tags": [category, "dine_in"],
                    "rating": 4.5,
                    "queue_time_min": 10,
                },
            ],
            "route": {"total_distance_km": 3.2, "total_travel_time_min": 25},
            "budget": {"total_price": 320},
            "availability": {"all_available": True, "max_queue_time_min": 10},
            "estimated_duration_min": 240,
            "tags": ["indoor", "low_intensity", category],
        },
        "weighted_score": score,
        "objective_vector": {"risk": 0.1, "route": 0.8, "group_fit": 0.8},
        "score_breakdown": {"risk": 0.02},
        "risk_factors": [],
    }


def _optimizer_plan(plan_id: str, *, restaurant_category: str, distance_km: float, price: int) -> dict:
    return {
        "plan_id": plan_id,
        "nodes": [
            {
                "poi_id": f"act_{plan_id}",
                "type": "activity",
                "name": f"Indoor Activity {plan_id}",
                "category": "indoor_activity",
                "tags": ["indoor", "low_intensity", "kid_friendly"],
                "price": 120,
                "duration_min": 90,
                "rating": 4.6,
                "queue_time_min": 8,
                "available_slots": [{"time": "14:00", "inventory_left": 20}],
                "product_ids": [f"prod_act_{plan_id}"],
                "merchant_id": f"m_act_{plan_id}",
            },
            {
                "poi_id": f"res_{plan_id}",
                "type": "restaurant",
                "name": f"Restaurant {plan_id}",
                "restaurant_category": restaurant_category,
                "tags": [restaurant_category, "dine_in", "family_friendly"],
                "price": price - 120,
                "duration_min": 70,
                "rating": 4.5,
                "queue_time_min": 10,
                "dine_in_available": True,
                "available_slots": [{"time": "16:30", "inventory_left": 20}],
                "product_ids": [f"prod_res_{plan_id}"],
                "merchant_id": f"m_res_{plan_id}",
            },
        ],
        "route": {"total_distance_km": distance_km, "total_travel_time_min": int(distance_km * 8)},
        "budget": {"total_price": price},
        "availability": {"all_available": True, "max_queue_time_min": 10},
        "estimated_duration_min": 250,
        "tags": ["indoor", "low_intensity", restaurant_category, "dine_in"],
        "schedule": {"activity_start": "14:00", "activity_end": "15:30", "restaurant_start": "16:30"},
    }


def test_plan_critic_is_default_off(monkeypatch):
    _clear_env(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("plan critic should be default off")

    monkeypatch.setattr(b_ai_plan_critic, "chat_completion", fail_if_called)
    candidates = [_candidate("cand_a", 0.7), _candidate("cand_b", 0.68)]

    updated, metadata = b_ai_plan_critic.apply_b_plan_critic(
        {"user_input": "family afternoon"},
        candidates,
        weights={"risk": -0.2},
    )

    assert updated == candidates
    assert metadata is None


def test_plan_critic_applies_bounded_top_k_rerank(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setenv("WF_B_AI_PLAN_CRITIC_ALWAYS", "1")

    def fake_chat_completion(messages, *, config):
        payload = json.loads(messages[1]["content"])
        assert len(payload["candidates"]) == 2
        return {
            "content": json.dumps(
                {
                    "candidate_adjustments": [
                        {
                            "plan_id": "cand_b",
                            "score_delta": 0.2,
                            "risk_delta": -0.1,
                            "confidence": 1.0,
                            "reasons": ["better diet fit for the spouse"],
                            "evidence": ["restaurant tags include light_food"],
                        }
                    ],
                    "global_notes": ["checked only top candidates"],
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 55},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_ai_plan_critic, "chat_completion", fake_chat_completion)
    candidates = [
        _candidate("cand_a", 0.7, category="hotpot"),
        _candidate("cand_b", 0.68, category="light_food"),
    ]

    updated, metadata = b_ai_plan_critic.apply_b_plan_critic(
        {
            "user_input": "Take my wife and child out; wife is dieting.",
            "scene_type": "family",
            "constraints": {"people_count": 3},
        },
        candidates,
        weights={"risk": -0.2},
    )

    assert updated[0]["plan"]["plan_id"] == "cand_b"
    assert updated[0]["weighted_score"] == 0.723
    assert updated[0]["objective_vector"]["risk"] == 0.06
    assert updated[0]["ai_critic_adjustment"]["score_delta"] == 0.035
    assert metadata["success"] is True
    assert metadata["applied_adjustments"][0]["plan_id"] == "cand_b"
    assert "test-key" not in str(metadata)


def test_plan_critic_skips_clear_low_risk_winner(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("plan critic should skip unambiguous deterministic ranking")

    monkeypatch.setattr(b_ai_plan_critic, "chat_completion", fail_if_called)
    candidates = [
        _candidate("cand_a", 0.9, category="light_food"),
        _candidate("cand_b", 0.78, category="hotpot"),
    ]

    updated, metadata = b_ai_plan_critic.apply_b_plan_critic(
        {
            "user_input": "今天下午带孩子轻松玩一下",
            "scene_type": "family",
            "constraints": {"people_count": 3},
        },
        candidates,
        weights={"risk": -0.2},
    )

    assert updated == candidates
    assert metadata["skipped"] is True
    assert metadata["reason"] == "clear_deterministic_winner"
    assert metadata["gate"]["score_gap"] == 0.12


def test_plan_critic_fallback_redacts_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setenv("WF_B_AI_PLAN_CRITIC_ALWAYS", "1")

    def fake_chat_completion(messages, *, config):
        raise RuntimeError("bad key test-key")

    monkeypatch.setattr(b_ai_plan_critic, "chat_completion", fake_chat_completion)

    updated, metadata = b_ai_plan_critic.apply_b_plan_critic(
        {"user_input": "hello"},
        [_candidate("cand_a", 0.7)],
        weights={"risk": -0.2},
    )

    assert updated[0]["plan"]["plan_id"] == "cand_a"
    assert metadata["success"] is False
    assert metadata["fallback"] is True
    assert "test-key" not in str(metadata)


def test_plan_optimizer_uses_critic_reranked_candidates(monkeypatch):
    def fake_apply_b_plan_critic(state, scored_candidates, *, weights):
        updated = [dict(item) for item in scored_candidates]
        updated.reverse()
        updated[0]["weighted_score"] = round(updated[0]["weighted_score"] + 0.03, 4)
        return updated, {
            "enabled": True,
            "provider": "test",
            "success": True,
            "selected_after_critic": updated[0]["plan"]["plan_id"],
            "applied_adjustments": [{"plan_id": updated[0]["plan"]["plan_id"], "score_delta": 0.03}],
        }

    monkeypatch.setattr(plan_optimizer_module, "apply_b_plan_critic", fake_apply_b_plan_critic)

    result = plan_optimizer_module.plan_optimizer_node(
        {
            "user_input": "今天下午带孩子轻松玩一下，再吃饭。",
            "scene_type": "family",
            "constraints": {
                "people_count": 3,
                "budget": 600,
                "max_distance_km": 10,
                "max_queue_time_min": 30,
                "duration_range_min": [180, 420],
                "child_age": 5,
            },
            "user_profile": {},
            "scenario_activities": ["亲子", "室内"],
            "filtered_candidates": [
                _optimizer_plan("cand_rule_first", restaurant_category="light_food", distance_km=2.0, price=330),
                _optimizer_plan("cand_critic_first", restaurant_category="family_restaurant", distance_km=3.0, price=360),
            ],
            "execution_log": [],
        }
    )

    assert result["selected_plan"]["plan_id"] == "plan_critic_first"
    assert result["selected_plan"]["b_ai_plan_critic"]["success"] is True
    assert result["b_ai_plan_critic"]["selected_after_critic"] == "cand_critic_first"
