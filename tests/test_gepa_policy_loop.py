from __future__ import annotations

import pytest

from experiments.gepa_policy_loop import compare_reports, guardrail_decision, parse_jsonish


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
