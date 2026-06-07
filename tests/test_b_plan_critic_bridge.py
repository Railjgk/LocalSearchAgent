from __future__ import annotations

from src.nodes.b_plan_critic_bridge import (
    build_ai_planning_review,
    build_b_replan_request,
)


def test_build_ai_planning_review_ignores_disabled_critic() -> None:
    assert build_ai_planning_review({"plan": {"plan_id": "p1"}}, {"enabled": False}) is None
    assert build_ai_planning_review({"plan": {"plan_id": "p1"}}, None) is None


def test_build_ai_planning_review_marks_replan_from_structural_terms() -> None:
    review = build_ai_planning_review(
        {"plan": {"plan_id": "p1"}},
        {
            "enabled": True,
            "success": True,
            "provider": "longcat",
            "model": "LongCat-Flash-Chat",
            "selected_after_critic": "p1",
            "global_notes": ["structural mismatch; re-search activity candidates"],
            "applied_adjustments": [
                {
                    "plan_id": "p1",
                    "score_delta": -0.01,
                    "risk_delta": 0.01,
                    "confidence": 0.4,
                    "reasons": ["activity and restaurant feel incoherent"],
                    "evidence": ["weak itinerary coherence"],
                }
            ],
        },
    )

    assert review is not None
    assert review["needs_replan"] is True
    assert review["replan_trigger"]["term_trigger"] is True
    assert review["selected_adjustment"]["plan_id"] == "p1"
    assert review["replan_guidance"] == ["activity and restaurant feel incoherent"]


def test_build_ai_planning_review_marks_replan_from_strong_negative_adjustment() -> None:
    review = build_ai_planning_review(
        {"plan": {"plan_id": "p1"}},
        {
            "enabled": True,
            "success": True,
            "global_notes": ["minor concern"],
            "applied_adjustments": [
                {"plan_id": "p1", "score_delta": -0.03, "risk_delta": 0.04, "confidence": 0.8}
            ],
        },
    )

    assert review is not None
    assert review["needs_replan"] is True
    assert review["replan_trigger"]["strong_negative_adjustment"] is True


def test_build_b_replan_request_targets_replacement_domains_and_preserves_constraints() -> None:
    review = {
        "needs_replan": True,
        "replan_guidance": ["replace restaurant because queue risk is high"],
        "global_notes": [],
        "replan_trigger": {"term_trigger": True},
    }
    selected_plan = {
        "plan_id": "frontend_plan",
        "scene_type": "family",
        "people_count": 3,
        "supply_identity": {"activity_id": "act_1", "restaurant_id": "res_1"},
    }
    plan_base = {"plan_id": "candidate_plan"}
    constraints = {
        "budget": 600,
        "max_distance_km": 8,
        "max_queue_time_min": 20,
        "duration_range": [180, 360],
        "planning_preferences": {"soft": ["少排队"]},
        "b_requirement_contract": {"version": "test"},
    }

    request = build_b_replan_request(selected_plan, plan_base, review, constraints)

    assert request is not None
    assert request["next_step"] == "rerun_candidate_generation"
    assert request["preserve_constraints"]["budget"] == 600
    assert request["candidate_generation_hints"]["avoid_plan_ids"] == ["candidate_plan"]
    assert request["candidate_generation_hints"]["avoid_supply_identity"]["restaurant_id"] == "res_1"
    assert request["rag_request"]["expected_output_key"] == "b_rag_candidate_evidence"
    assert request["rag_request"]["target_nodes"] == [
        {
            "supply_domain": "restaurant",
            "query_terms": ["replace restaurant because queue risk is high"],
            "avoid_poi_ids": ["res_1"],
            "max_candidates": 12,
        }
    ]


def test_build_b_replan_request_returns_none_without_replan_signal() -> None:
    assert build_b_replan_request({}, {}, {"needs_replan": False}, {}) is None
