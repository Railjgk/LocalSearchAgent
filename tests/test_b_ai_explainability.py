from __future__ import annotations

import json

from src.nodes import explainability


def _selected_plan_state() -> dict:
    return {
        "user_input": "family afternoon plan",
        "scene_type": "family",
        "constraints": {
            "budget": 400,
            "people_count": 3,
            "child_age": 5,
            "mom_diet": "low_calorie",
            "max_distance_km": 8,
            "max_queue_time": 30,
        },
        "user_profile": {"avoid": ["long_queue"], "food_preference": ["light_food"]},
        "selected_plan": {
            "title": "Family afternoon plan",
            "timeline": [
                {
                    "type": "play",
                    "activity": "Indoor Playground",
                    "time": "14:30-16:00",
                    "duration_min": 90,
                    "price": 120,
                    "notes": ["kid_friendly", "low_intensity"],
                },
                {
                    "type": "transition",
                    "activity": "Walk",
                    "time": "16:00-16:20",
                    "duration_min": 20,
                },
                {
                    "type": "restaurant",
                    "activity": "Light Cafe",
                    "time": "17:00-18:00",
                    "duration_min": 60,
                    "price": 180,
                    "notes": ["low_oil_low_salt"],
                },
            ],
            "total_price": 300,
            "total_duration_min": 170,
            "total_distance_km": 4.2,
            "objective_vector": {
                "preference": 0.8,
                "group_fit": 0.9,
                "route": 0.8,
                "budget": 0.9,
                "availability": 0.8,
                "experience": 0.7,
                "risk": 0.1,
            },
            "score_breakdown": {"route": 0.16, "budget": 0.18},
            "risk_factors": [],
            "constraint_summary": {
                "distance_status": "\u2713",
                "queue_status": "\u2713",
                "budget_status": "\u2713",
                "child_friendly_status": "\u2713",
                "diet_status": "\u2713",
            },
            "execution_ready": True,
        },
        "optimization_score": 88.8,
        "alternative_plans": [
            {
                "title": "Shorter route option",
                "dominant_dimension": "route",
                "tradeoff": "shorter travel but lower experience",
                "total_price": 330,
                "total_distance_km": 3.1,
                "objective_vector": {"route": 0.95, "experience": 0.55},
            }
        ],
        "execution_log": [],
    }


def _clear_longcat_env(monkeypatch):
    for key in ("WF_B_AI_ENABLED", "LONGCAT_API_KEY", "LONGCAT_APP_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_explainability_does_not_call_ai_by_default(monkeypatch):
    _clear_longcat_env(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AI should be default off")

    monkeypatch.setattr(explainability, "chat_completion", fail_if_called)

    result = explainability.explainability_node(_selected_plan_state())

    assert result["explanation_text"]
    assert "b_ai_explanation" not in result


def test_explainability_uses_longcat_when_enabled(monkeypatch):
    _clear_longcat_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    captured = {}
    ai_text = "\u8fd9\u662fAI\u7248\u63a8\u8350\u7406\u7531"

    def fake_chat_completion(messages, *, config):
        captured["messages"] = messages
        captured["config"] = config
        return {
            "content": json.dumps(
                {
                    "explanation_text": ai_text,
                    "risk_notes": ["confirm slot"],
                    "next_best_action": "reserve first",
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 42},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(explainability, "chat_completion", fake_chat_completion)

    result = explainability.explainability_node(_selected_plan_state())

    payload = json.loads(captured["messages"][1]["content"])
    assert payload["selected_plan"]["title"] == "Family afternoon plan"
    assert payload["deterministic_explanation"]
    assert result["explanation_text"] == ai_text
    assert result["b_ai_explanation"]["success"] is True
    assert result["b_ai_explanation"]["provider"] == "longcat"
    assert result["b_ai_explanation"]["usage"] == {"total_tokens": 42}


def test_explainability_falls_back_when_longcat_fails(monkeypatch):
    _clear_longcat_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        raise RuntimeError("provider failed for test-key")

    monkeypatch.setattr(explainability, "chat_completion", fake_chat_completion)

    result = explainability.explainability_node(_selected_plan_state())

    assert result["explanation_text"]
    assert result["b_ai_explanation"]["success"] is False
    assert result["b_ai_explanation"]["fallback"] is True
    assert "test-key" not in str(result["b_ai_explanation"])
