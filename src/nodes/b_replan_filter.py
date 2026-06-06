"""Candidate filtering helpers for bounded B replanning."""
from __future__ import annotations

from typing import Any


def active_replan_request(state: dict | None, constraints: dict | None) -> dict:
    state = state or {}
    constraints = constraints or {}
    request = state.get("b_replan_request") or constraints.get("b_replan_request") or {}
    return request if isinstance(request, dict) else {}


def replan_avoid_identity_matches(plan: dict, request: dict) -> bool:
    hints = request.get("candidate_generation_hints") or {}
    avoid_plan_ids = {str(item) for item in (hints.get("avoid_plan_ids") or []) if item}
    if str(plan.get("plan_id") or "") in avoid_plan_ids:
        return True

    identity = hints.get("avoid_supply_identity") or {}
    if not isinstance(identity, dict):
        return False
    nodes = plan.get("nodes", []) or []
    activity = next((node for node in nodes if node.get("type") == "activity"), {})
    restaurant = next((node for node in nodes if node.get("type") == "restaurant"), {})
    activity_id = identity.get("activity_id")
    restaurant_id = identity.get("restaurant_id")
    if activity_id and restaurant_id:
        return activity.get("poi_id") == activity_id and restaurant.get("poi_id") == restaurant_id
    if activity_id and activity.get("poi_id") == activity_id:
        return True
    if restaurant_id and restaurant.get("poi_id") == restaurant_id:
        return True
    return False


def filter_replan_avoided_plans(plan_candidates: list[dict], request: dict) -> tuple[list[dict], dict[str, Any]]:
    if not request:
        return plan_candidates, {"applied": False}
    kept = [plan for plan in plan_candidates if not replan_avoid_identity_matches(plan, request)]
    removed = len(plan_candidates) - len(kept)
    if removed <= 0 or not kept:
        return plan_candidates, {"applied": False, "removed": removed, "kept": len(kept)}
    return kept, {
        "applied": True,
        "removed": removed,
        "kept": len(kept),
        "source": request.get("source"),
        "status": request.get("status"),
    }
