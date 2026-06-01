from __future__ import annotations

from src.graph import WORKFLOW_NODES
from src.nodes.b_ai_trace import b_ai_trace_node, build_b_ai_trace


def test_b_ai_trace_is_noop_without_ai_metadata():
    assert b_ai_trace_node({"execution_log": []}) == {}


def test_b_ai_trace_aggregates_stage_metadata_and_redacts_key(monkeypatch):
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    state = {
        "b_ai_semantic_hints": {
            "enabled": True,
            "provider": "longcat",
            "model": "LongCat-Flash-Chat",
            "success": True,
            "usage": {"total_tokens": 20},
            "hints": {
                "confidence": 0.81,
                "soft_tags": ["亲子"],
                "avoid_tags": ["排队久"],
                "activity_intent_tags": ["手作"],
                "restaurant_intent_tags": ["轻食", "健康餐"],
            },
        },
        "b_ai_plan_critic": {
            "enabled": True,
            "provider": "longcat",
            "model": "LongCat-Flash-Chat",
            "success": True,
            "usage": {"total_tokens": 34},
            "guardrails": {"top_k": 5, "max_score_delta": 0.035, "max_risk_delta": 0.08},
            "selected_after_critic": "plan_002",
            "applied_adjustments": [{"plan_id": "plan_002", "score_delta": 0.02}],
        },
        "b_ai_explanation": {
            "enabled": True,
            "provider": "longcat",
            "success": False,
            "fallback": True,
            "error": "bad key test-key",
            "error_type": "RuntimeError",
        },
        "b_ai_repair_plan": {
            "enabled": True,
            "provider": "longcat",
            "model": "LongCat-Flash-Chat",
            "success": True,
            "usage": {"total_tokens": 18},
            "repair_strategy": "retry_same_poi_new_slot",
        },
        "b_repair_plan": {"repair_strategy": "retry_same_poi_new_slot"},
        "execution_log": [],
    }

    result = b_ai_trace_node(state)
    trace = result["b_ai_trace"]

    assert trace["enabled_stage_count"] == 4
    assert trace["total_tokens"] == 72
    assert trace["successful_stages"] == ["semantic_hints", "plan_critic", "repair_planner"]
    assert trace["fallback_stages"] == ["explanation"]
    assert trace["decision_impact"]["selected_after_critic"] == "plan_002"
    assert trace["decision_impact"]["repair_strategy"] == "retry_same_poi_new_slot"
    assert trace["stages"][0]["hint_counts"]["restaurant_intent_tags"] == 2
    assert trace["stages"][1]["adjustment_count"] == 1
    assert "test-key" not in str(trace)
    assert result["execution_log"][-1].startswith("[B] b_ai_trace_node recorded")


def test_build_b_ai_trace_handles_partial_metadata():
    trace = build_b_ai_trace(
        {
            "b_ai_plan_critic": {
                "enabled": True,
                "provider": "test",
                "success": True,
                "usage": {"total_token_count": 9},
            }
        }
    )

    assert trace is not None
    assert trace["enabled_stage_count"] == 1
    assert trace["total_tokens"] == 9
    assert trace["stages"][0]["stage"] == "plan_critic"


def test_graph_records_trace_after_repair_planner_before_payment():
    node_names = [node.__name__ for node in WORKFLOW_NODES]

    assert node_names.index("repair_planner_node") < node_names.index("b_ai_trace_node")
    assert node_names.index("b_ai_trace_node") < node_names.index("payment_layer_node")
