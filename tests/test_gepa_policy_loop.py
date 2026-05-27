from __future__ import annotations

import pytest

from experiments.gepa_policy_loop import (
    build_trace,
    compare_reports,
    guardrail_decision,
    parse_jsonish,
)
from experiments.reflect_and_mutate import apply_reflection_to_policy, validate_policy_proposal


def test_parse_jsonish_accepts_fenced_json():
    payload = parse_jsonish(
        """```json
{"reflection_summary": {"overall_judgment": "ok"}, "policy_change_proposals": []}
```"""
    )

    assert payload["reflection_summary"]["overall_judgment"] == "ok"
    assert payload["policy_change_proposals"] == []


def test_parse_jsonish_rejects_non_json():
    with pytest.raises(ValueError):
        parse_jsonish("not json")


def test_compare_reports_detects_regressions_and_changed_selection():
    baseline = {
        "aggregate": {
            "pass_rate": 100.0,
            "avg_optimization_score": 80.0,
            "execution_ready_rate": 100.0,
        },
        "cases": [
            {"case_id": "a", "passed": True, "selected_plan_id": "plan_1"},
            {"case_id": "b", "passed": False, "selected_plan_id": None},
        ],
    }
    mutated = {
        "aggregate": {
            "pass_rate": 50.0,
            "avg_optimization_score": 81.5,
            "execution_ready_rate": 100.0,
        },
        "cases": [
            {"case_id": "a", "passed": False, "selected_plan_id": "plan_2"},
            {"case_id": "b", "passed": True, "selected_plan_id": "plan_3"},
        ],
    }

    comparison = compare_reports(baseline, mutated)

    assert comparison["delta"]["pass_rate"] == -50.0
    assert comparison["case_changes"]["improved"] == ["b"]
    assert comparison["case_changes"]["regressed"] == ["a"]
    assert comparison["case_changes"]["changed_selection"] == ["a", "b"]


def test_compare_reports_includes_quality_metric_delta():
    baseline = {
        "aggregate": {
            "total_cases": 1,
            "pass_rate": 100.0,
            "avg_optimization_score": 80.0,
            "execution_ready_rate": 100.0,
        },
        "cases": [{"case_id": "a", "passed": True, "selected_plan_id": "plan_1"}],
        "traces": [
            {
                "selected_plan": {
                    "objective_vector": {"group_fit": 0.9, "route": 0.8},
                    "plan_quality": {"fulfillment_confidence": 0.9},
                    "why_selected": {"evidence": ["good"], "tradeoffs": []},
                }
            }
        ],
    }
    mutated = {
        "aggregate": {
            "total_cases": 1,
            "pass_rate": 100.0,
            "avg_optimization_score": 81.0,
            "execution_ready_rate": 100.0,
        },
        "cases": [{"case_id": "a", "passed": True, "selected_plan_id": "plan_1"}],
        "traces": [
            {
                "selected_plan": {
                    "objective_vector": {"group_fit": 0.82, "route": 0.81},
                    "plan_quality": {"fulfillment_confidence": 0.88},
                    "why_selected": {"evidence": ["good", "better"], "tradeoffs": ["risk"]},
                }
            }
        ],
    }

    comparison = compare_reports(baseline, mutated)

    assert comparison["quality"]["delta"]["objective_averages"]["group_fit"] == -0.08
    assert comparison["quality"]["delta"]["objective_averages"]["route"] == 0.01
    assert comparison["quality"]["delta"]["avg_evidence_items"] == 1.0


def test_build_trace_preserves_plan_quality_evidence():
    trace = build_trace(
        {"case_id": "family_case"},
        {
            "selected_plan": {
                "plan_id": "plan_a",
                "objective_vector": {"group_fit": 0.9},
                "plan_quality": {"family_facility_fit": 0.88},
                "quality_adjustments": [{"dimension": "group_fit"}],
                "why_selected": {"evidence": ["餐厅支持宝宝椅"]},
            }
        },
        [],
    )

    assert trace["selected_plan_quality"]["family_facility_fit"] == 0.88
    assert trace["selected_quality_adjustments"][0]["dimension"] == "group_fit"
    assert trace["selected_why_selected"]["evidence"] == ["餐厅支持宝宝椅"]


def test_guardrail_rejects_regressed_cases():
    decision = guardrail_decision(
        {
            "mutated": {"pass_rate": 100.0, "execution_ready_rate": 100.0},
            "delta": {"avg_optimization_score": 0.0},
            "case_changes": {"regressed": ["case_a"]},
        },
        min_pass_rate=100.0,
        min_execution_ready_rate=100.0,
        allow_avg_score_drop=1.0,
    )

    assert decision["accepted"] is False
    assert "regressed cases" in decision["reasons"][0]


def test_guardrail_accepts_no_regression_with_small_score_drop():
    decision = guardrail_decision(
        {
            "mutated": {"pass_rate": 100.0, "execution_ready_rate": 100.0},
            "delta": {"avg_optimization_score": -0.5},
            "case_changes": {"regressed": []},
        },
        min_pass_rate=100.0,
        min_execution_ready_rate=100.0,
        allow_avg_score_drop=1.0,
    )

    assert decision["accepted"] is True
    assert decision["reasons"] == []


def test_guardrail_rejects_protected_quality_drop():
    decision = guardrail_decision(
        {
            "mutated": {"total_cases": 1, "pass_rate": 100.0, "execution_ready_rate": 100.0},
            "delta": {"avg_optimization_score": 0.0},
            "case_changes": {"regressed": [], "changed_selection": []},
            "quality": {"delta": {"objective_averages": {"group_fit": -0.08}}},
        },
        min_pass_rate=100.0,
        min_execution_ready_rate=100.0,
        allow_avg_score_drop=1.0,
        allow_quality_metric_drop=0.05,
    )

    assert decision["accepted"] is False
    assert "protected objective group_fit" in decision["reasons"][0]


def test_policy_proposal_rejects_non_policy_paths():
    proposal = {
        "target_area": "code",
        "proposed_change": {
            "type": "adjust_value",
            "path": "src.nodes.plan_optimizer.score",
            "new_value": 0.1,
        },
    }

    problems = validate_policy_proposal(proposal)

    assert "target_area not allowed: code" in problems
    assert "path not allowed: src.nodes.plan_optimizer.score" in problems


def test_apply_reflection_skips_unsafe_policy_changes():
    policy = {
        "policy_name": "baseline",
        "version": "0.1",
        "scene_weights": {"family": {"group_fit": 0.25}},
    }
    reflection = {
        "policy_change_proposals": [
            {
                "change_id": "safe",
                "priority": "high",
                "target_area": "scene_weights",
                "proposed_change": {
                    "type": "adjust_value",
                    "path": "scene_weights.family.group_fit",
                    "new_value": 0.3,
                },
            },
            {
                "change_id": "unsafe",
                "priority": "high",
                "target_area": "code",
                "proposed_change": {
                    "type": "adjust_value",
                    "path": "src.nodes.plan_optimizer.score",
                    "new_value": 0.1,
                },
            },
        ]
    }

    mutated, applied = apply_reflection_to_policy(policy, reflection, max_changes=5)

    assert mutated["scene_weights"]["family"]["group_fit"] == 0.3
    assert [item["change_id"] for item in applied] == ["safe"]
    assert mutated["mutation_meta"]["rejected_change_count"] == 1
    assert mutated["mutation_meta"]["rejected_changes"][0]["change_id"] == "unsafe"
