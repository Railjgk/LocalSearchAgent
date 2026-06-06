from __future__ import annotations

from src.nodes.b_replan_filter import (
    active_replan_request,
    filter_replan_avoided_plans,
    replan_avoid_identity_matches,
)


def _plan(plan_id: str, activity_id: str = "act_1", restaurant_id: str = "res_1") -> dict:
    return {
        "plan_id": plan_id,
        "nodes": [
            {"type": "activity", "poi_id": activity_id},
            {"type": "restaurant", "poi_id": restaurant_id},
        ],
    }


def test_active_replan_request_prefers_state_over_constraints() -> None:
    assert active_replan_request(
        {"b_replan_request": {"source": "state"}},
        {"b_replan_request": {"source": "constraints"}},
    ) == {"source": "state"}


def test_active_replan_request_ignores_non_dict_payloads() -> None:
    assert active_replan_request({"b_replan_request": "bad"}, {}) == {}


def test_replan_avoid_identity_matches_plan_id() -> None:
    request = {"candidate_generation_hints": {"avoid_plan_ids": ["old_plan"]}}

    assert replan_avoid_identity_matches(_plan("old_plan"), request) is True
    assert replan_avoid_identity_matches(_plan("new_plan"), request) is False


def test_replan_avoid_identity_matches_supply_identity_pair() -> None:
    request = {
        "candidate_generation_hints": {
            "avoid_supply_identity": {"activity_id": "act_1", "restaurant_id": "res_1"}
        }
    }

    assert replan_avoid_identity_matches(_plan("same"), request) is True
    assert replan_avoid_identity_matches(_plan("different", restaurant_id="res_2"), request) is False


def test_filter_replan_avoided_plans_keeps_original_when_all_would_be_removed() -> None:
    request = {"candidate_generation_hints": {"avoid_plan_ids": ["only_plan"]}}
    plans = [_plan("only_plan")]

    kept, metadata = filter_replan_avoided_plans(plans, request)

    assert kept == plans
    assert metadata == {"applied": False, "removed": 1, "kept": 0}


def test_filter_replan_avoided_plans_reports_applied_filter() -> None:
    request = {
        "source": "longcat_plan_critic",
        "status": "needs_replan",
        "candidate_generation_hints": {"avoid_plan_ids": ["old_plan"]},
    }
    kept, metadata = filter_replan_avoided_plans([_plan("old_plan"), _plan("new_plan")], request)

    assert [plan["plan_id"] for plan in kept] == ["new_plan"]
    assert metadata == {
        "applied": True,
        "removed": 1,
        "kept": 1,
        "source": "longcat_plan_critic",
        "status": "needs_replan",
    }
