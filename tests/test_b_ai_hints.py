from __future__ import annotations

import json

from src.nodes import b_ai_hints


def _clear_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "WF_B_AI_SEMANTIC_HINTS_ENABLED",
        "WF_B_AI_PLAN_CRITIC_ENABLED",
        "WF_B_AI_REPAIR_PLANNER_ENABLED",
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
                    "restaurant_intent_tags": ["coffee"],
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
            "user_input": (
                "\u60f3\u5728\u4e0a\u6d77\u505a\u4e00\u70b9\u672c\u5730"
                "\u751f\u6d3b\u4f53\u9a8c\uff0c\u522b\u53bb\u592a\u6324\u7684\u5546\u573a"
            ),
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
    assert "coffee" in constraints["planning_preferences"]["food_type"]
    assert "crowded_mall" in constraints["avoid"]
    assert "crowded_mall" in user_profile["avoid"]
    assert "local_market" in scenario_activities
    assert metadata["success"] is True
    assert metadata["hints"]["confidence"] == 0.82


def test_semantic_hints_do_not_infer_diet_from_coffee_request(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "soft_tags": ["quiet", "low_calorie"],
                    "restaurant_intent_tags": ["coffee", "light_food", "low_sugar"],
                    "confidence": 0.91,
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 24},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_ai_hints, "chat_completion", fake_chat_completion)

    constraints, _, _, metadata = b_ai_hints.apply_b_semantic_hints(
        {
            "user_input": (
                "\u4eca\u5929\u60f3\u4e00\u4e2a\u4eba\u770b\u5c55\uff0c"
                "\u7136\u540e\u627e\u5b89\u9759\u5496\u5561\u5e97\u5750\u4e00\u4f1a\uff0c"
                "\u4e0d\u60f3\u5403\u6b63\u9910"
            ),
            "constraints": {"planning_preferences": {"food_type": ["\u5496\u5561"]}},
        },
        constraints={"planning_preferences": {"food_type": ["\u5496\u5561"]}},
        user_profile={},
        scenario_activities=[],
    )

    assert "quiet" in constraints["soft_tags"]
    assert "coffee" in constraints["planning_preferences"]["food_type"]
    assert "low_calorie" not in constraints["soft_tags"]
    assert "light_food" not in constraints["planning_preferences"]["food_type"]
    assert "low_sugar" not in constraints["planning_preferences"]["food_type"]
    assert metadata["hints"]["restaurant_intent_tags"] == ["coffee"]


def test_semantic_hints_keep_diet_when_user_says_slimming(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "soft_tags": ["low_calorie"],
                    "restaurant_intent_tags": ["light_food", "low_sugar"],
                    "confidence": 0.88,
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 25},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_ai_hints, "chat_completion", fake_chat_completion)

    constraints, _, _, metadata = b_ai_hints.apply_b_semantic_hints(
        {
            "user_input": "\u8001\u5a46\u6700\u8fd1\u5728\u51cf\u8102\uff0c\u665a\u9910\u60f3\u6e05\u6de1\u4f4e\u5361",
            "constraints": {"planning_preferences": {"food_type": []}},
        },
        constraints={"planning_preferences": {"food_type": []}},
        user_profile={},
        scenario_activities=[],
    )

    assert "low_calorie" in constraints["soft_tags"]
    assert "light_food" in constraints["planning_preferences"]["food_type"]
    assert "low_sugar" in constraints["planning_preferences"]["food_type"]
    assert metadata["success"] is True


def test_semantic_hints_do_not_self_confirm_from_existing_english_light_food(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        return {
            "content": json.dumps(
                {
                    "soft_tags": ["local_culture", "light_food"],
                    "restaurant_intent_tags": ["light_food"],
                    "confidence": 0.7,
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 20},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(b_ai_hints, "chat_completion", fake_chat_completion)

    constraints, _, _, metadata = b_ai_hints.apply_b_semantic_hints(
        {
            "user_input": (
                "\u770b\u5c55\u540e\u5728\u9644\u8fd1\u5403\u4e2a\u4e2d\u996d\uff0c"
                "\u518d\u4e70\u70b9\u672c\u5730\u7279\u4ea7"
            ),
            "constraints": {"soft_tags": ["light_food"]},
        },
        constraints={"soft_tags": ["light_food"], "planning_preferences": {"food_type": []}},
        user_profile={},
        scenario_activities=[],
    )

    assert "local_culture" in constraints["soft_tags"]
    assert "light_food" not in constraints["planning_preferences"]["food_type"]
    assert metadata["hints"]["restaurant_intent_tags"] == []


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
