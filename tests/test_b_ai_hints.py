from __future__ import annotations

import json

from src.nodes import b_ai_hints


def _clear_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "WF_B_AI_SEMANTIC_HINTS_ENABLED",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_semantic_hints_are_default_off(monkeypatch):
    _clear_env(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("semantic hints should be default off")

    monkeypatch.setattr(b_ai_hints, "chat_completion", fail_if_called)

    constraints, user_profile, scenario_activities, metadata = b_ai_hints.apply_b_semantic_hints(
        {"user_input": "want citywalk"},
        constraints={},
        user_profile={},
        scenario_activities=[],
    )

    assert constraints == {}
    assert user_profile == {}
    assert scenario_activities == []
    assert metadata is None


def test_semantic_hints_merge_only_soft_fields(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "soft_tags": ["local_culture", "hallucinated_tag"],
                    "avoid_tags": ["crowded_mall"],
                    "activity_intent_tags": ["citywalk", "local_market"],
                    "restaurant_intent_tags": ["light_food", "vegetable_rich"],
                    "route_priority": "nearby",
                    "budget_priority": "balanced",
                    "confidence": 0.82,
                    "evidence": ["wants local weekend flow"],
                    "hard_constraints": {"budget": 99},
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 30},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_ai_hints, "chat_completion", fake_chat_completion)

    constraints, user_profile, scenario_activities, metadata = b_ai_hints.apply_b_semantic_hints(
        {
            "user_input": "想在上海做一点本地生活体验，别去太挤的商场",
            "constraints": {"budget": 400},
            "user_profile": {},
            "scenario_activities": [],
        },
        constraints={"budget": 400, "soft_tags": ["nearby"]},
        user_profile={},
        scenario_activities=["nearby"],
    )

    assert constraints["budget"] == 400
    assert "local_culture" in constraints["soft_tags"]
    assert "hallucinated_tag" not in constraints["soft_tags"]
    assert "citywalk" in constraints["planning_preferences"]["activity_type"]
    assert "light_food" in constraints["planning_preferences"]["food_type"]
    assert "crowded_mall" in constraints["avoid"]
    assert "crowded_mall" in user_profile["avoid"]
    assert "local_market" in scenario_activities
    assert metadata["success"] is True
    assert metadata["hints"]["confidence"] == 0.82


def test_semantic_hints_fallback_redacts_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        raise RuntimeError("bad key test-key")

    monkeypatch.setattr(b_ai_hints, "chat_completion", fake_chat_completion)

    constraints, user_profile, scenario_activities, metadata = b_ai_hints.apply_b_semantic_hints(
        {"user_input": "hello"},
        constraints={},
        user_profile={},
        scenario_activities=[],
    )

    assert constraints == {}
    assert user_profile == {}
    assert scenario_activities == []
    assert metadata["success"] is False
    assert metadata["fallback"] is True
    assert "test-key" not in str(metadata)
